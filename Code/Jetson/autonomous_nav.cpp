// =============================================================================
//  autonomous_nav.cpp
//  Jetson Orin Nano - OpenDog V3 Autonomous Obstacle Avoidance
//
//  Hardware:
//    - IMX219-83 Stereo Camera on CAM0 (left) + CAM1 (right) via CSI
//    - NRF24L01 radio module:
//        CE  -> physical pin 29 (PQ.05 = gpiochip0 line 105)
//        CSN -> physical pin 24 (SPI0_CS0 = /dev/spidev0.0)
//        SCK -> physical pin 23
//        MOSI-> physical pin 19
//        MISO-> physical pin 21
//        VCC -> 3.3V (pin 1 or 17)
//        GND -> GND  (pin 6, 9, etc)
//
//  Kill switch: physical power switch on Jetson power supply.
//  Press Ctrl+C to stop the program cleanly.
//
//  Behaviour:
//    1. Streams stereo frames -> SGBM disparity -> depth map
//    2. Samples a central ROI for minimum depth
//    3. If obstacle closer than OBSTACLE_DIST_M:
//         - Determines turn direction from left/right depth difference
//         - Sends LT (twist) command for TURN_DURATION_MS
//         - Resumes straight walking
//    4. Walk forward ramps RFB from 512 down toward 0 over RAMP_DURATION_MS
//       RFB below centre = forward (robot brain applies * -1 to RFB)
//    5. Radio runs in a dedicated thread sending every 10ms so the
//       robot watchdog never fires even during heavy stereo processing
//
//  Stick mapping (raw 0-1023, centre = 512):
//    RFB  - forward/backward  (below 512 = forward, 512 = stop)
//    LT   - yaw twist         (512 = centre, <512 = left, >512 = right)
//    All other axes held at 512 (neutral)
//
//  mode field = 6 (walking mode, pre-selected by user on physical remote)
//
//  Build:
//    g++ autonomous_nav.cpp -o autonomous_nav \
//        $(pkg-config --cflags --libs opencv4) \
//        -lrf24 -lpthread -std=c++17 -O2
// =============================================================================

#include <RF24/RF24.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <string>
#include <thread>

// =============================================================================
// Tuneable parameters - adjust to taste
// =============================================================================

// Obstacle detection
static const float OBSTACLE_DIST_M = 0.34f; // metres - trigger turn if closer
static const float CLEAR_DIST_M =
    0.35f; // metres - resume walk when path this clear
static const int ROI_CENTER_FRAC = 4; // use centre 1/N of frame for depth ROI

// Timing (milliseconds)
static const int TURN_DURATION_MS = 1200; // how long to apply a turn command
static const int RAMP_DURATION_MS = 800;  // ramp-up time for forward walk
static const int RADIO_INTERVAL_MS =
    10; // packet send rate - 10ms for stable connection

// Stick raw values (0-1023 range, 512 = centre)
static const int16_t STICK_CENTER = 512;
static const int16_t STICK_TURN_LEFT = 50; // LT value for left yaw  (toward 0)
static const int16_t STICK_TURN_RIGHT =
    974; // LT value for right yaw (toward 1023)

// Camera settings
static const int CAM_WIDTH = 1640;
static const int CAM_HEIGHT = 1232;
static const int CAM_FPS = 30;
static const float BASELINE_M = 0.060f; // IMX219-83 baseline ~60 mm
static const float FOCAL_PX = 700.0f;   // approximate focal length

// NRF24L01 pins
// CE  -> physical pin 29 = PQ.05 = gpiochip0 line 105
// CSN -> physical pin 24 = SPI0_CS0 = /dev/spidev0.0
static const uint8_t RF24_CE_PIN = 105;
static const uint8_t RF24_CSN_PIN = 0;

// =============================================================================
// Packet structure - MUST exactly match openDogV3 RECEIVE_DATA_STRUCTURE
// =============================================================================
#pragma pack(push, 1)
struct SendDataStructure
{
    int16_t menuDown;
    int16_t Select;
    int16_t menuUp;
    int16_t toggleBottom;
    int16_t toggleTop;
    int16_t toggle1;
    int16_t toggle2;
    int16_t mode;
    int16_t RLR;
    int16_t RFB;
    int16_t RT;
    int16_t LLR;
    int16_t LFB;
    int16_t LT;
};
#pragma pack(pop)

// =============================================================================
// Robot state machine
// =============================================================================
enum class RobotState
{
    WALKING_FORWARD,
    AVOIDING_OBSTACLE
};

// =============================================================================
// Global state
// =============================================================================
std::atomic<bool> g_running(true);
SendDataStructure g_packet;
std::mutex g_packetMutex;

void signalHandler(int /*sig*/) { g_running = false; }

// =============================================================================
// Returns a fully neutral packet with mode=6 and motors armed
// =============================================================================
SendDataStructure neutralPacket()
{
    SendDataStructure p;
    p.menuDown = 0;
    p.Select = 0;
    p.menuUp = 0;
    p.toggleBottom = 0;
    p.toggleTop = 1;
    p.toggle1 = 0;
    p.toggle2 = 0;
    p.mode = 6;
    p.RLR = STICK_CENTER;
    p.RFB = STICK_CENTER;
    p.RT = STICK_CENTER;
    p.LLR = STICK_CENTER;
    p.LFB = STICK_CENTER;
    p.LT = STICK_CENTER;
    return p;
}

// =============================================================================
// Radio transmit thread - sends packets every RADIO_INTERVAL_MS
// =============================================================================
void radioThread(RF24 *radio)
{
    while (g_running)
    {
        {
            std::lock_guard<std::mutex> lock(g_packetMutex);
            radio->write(&g_packet, sizeof(SendDataStructure));
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(RADIO_INTERVAL_MS));
    }
}

// =============================================================================
// GStreamer pipeline for IMX219 on Jetson Orin
// =============================================================================
std::string gstPipeline(int sensorId)
{
    return std::string("nvarguscamerasrc sensor-id=") +
           std::to_string(sensorId) + " ! video/x-raw(memory:NVMM)" +
           ", width=" + std::to_string(CAM_WIDTH) +
           ", height=" + std::to_string(CAM_HEIGHT) +
           ", framerate=" + std::to_string(CAM_FPS) + "/1" +
           " ! nvvidconv flip-method=0" + " ! video/x-raw" +
           ", width=" + std::to_string(CAM_WIDTH) +
           ", height=" + std::to_string(CAM_HEIGHT) + ", format=BGRx" +
           " ! videoconvert" + " ! video/x-raw, format=BGR" +
           " ! appsink drop=true max-buffers=1 sync=false";
}

// =============================================================================
// StereoSGBM matcher
// =============================================================================
cv::Ptr<cv::StereoSGBM> createSGBM()
{
    const int minDisparity = 0;
    const int numDisparities = 128;
    const int blockSize = 5;
    const int P1 = 8 * 3 * blockSize * blockSize;
    const int P2 = 32 * 3 * blockSize * blockSize;
    const int disp12MaxDiff = 1;
    const int preFilterCap = 63;
    const int uniquenessRatio = 10;
    const int speckleWindowSize = 100;
    const int speckleRange = 32;
    const int mode = cv::StereoSGBM::MODE_SGBM_3WAY;

    return cv::StereoSGBM::create(
        minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff,
        preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, mode);
}

// =============================================================================
// Disparity to depth
// =============================================================================
cv::Mat disparityToDepth(const cv::Mat &disparity32F)
{
    cv::Mat depth(disparity32F.size(), CV_32F, cv::Scalar(0.0f));
    for (int r = 0; r < disparity32F.rows; ++r)
        for (int c = 0; c < disparity32F.cols; ++c)
        {
            float d = disparity32F.at<float>(r, c);
            if (d > 1.0f)
                depth.at<float>(r, c) = (FOCAL_PX * BASELINE_M) / d;
        }
    return depth;
}

// =============================================================================
// Depth sample from central ROI
// =============================================================================
struct DepthSample
{
    float center, left, right;
};

DepthSample sampleDepth(const cv::Mat &depthMap)
{
    int w = depthMap.cols;
    int h = depthMap.rows;

    int roiW = w / ROI_CENTER_FRAC;
    int roiH = h / ROI_CENTER_FRAC;
    int roiX = (w - roiW) / 2;
    int roiY = (h - roiH) / 2;

    cv::Rect centerROI(roiX, roiY, roiW, roiH);
    cv::Rect leftROI(roiX, roiY, roiW / 2, roiH);
    cv::Rect rightROI(roiX + roiW / 2, roiY, roiW / 2, roiH);

    auto minPositive = [](const cv::Mat &m) -> float
    {
        float mn = std::numeric_limits<float>::max();
        for (int r = 0; r < m.rows; ++r)
            for (int c = 0; c < m.cols; ++c)
            {
                float v = m.at<float>(r, c);
                if (v > 0.05f && v < mn)
                    mn = v;
            }
        return (mn == std::numeric_limits<float>::max()) ? 99.0f : mn;
    };

    DepthSample ds;
    ds.center = minPositive(depthMap(centerROI));
    ds.left = minPositive(depthMap(leftROI));
    ds.right = minPositive(depthMap(rightROI));
    return ds;
}

// =============================================================================
// main
// =============================================================================
int main()
{
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // -------------------------------------------------------------------------
    // Open cameras
    // -------------------------------------------------------------------------
    std::cout << "[INFO] Opening cameras...\n";
    cv::VideoCapture capLeft(gstPipeline(0), cv::CAP_GSTREAMER);
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    cv::VideoCapture capRight(gstPipeline(1), cv::CAP_GSTREAMER);

    if (!capLeft.isOpened() || !capRight.isOpened())
    {
        std::cerr
            << "[ERROR] Could not open one or both cameras.\n"
            << "        Check CSI ribbon connections and nvarguscamerasrc.\n";
        return 1;
    }
    std::cout << "[INFO] Both cameras open.\n";

    // -------------------------------------------------------------------------
    // Stereo matcher and WLS disparity filter
    // -------------------------------------------------------------------------
    cv::Ptr<cv::StereoSGBM> sgbm = createSGBM();
    cv::Ptr<cv::StereoMatcher> rightMatcher =
        cv::ximgproc::createRightMatcher(sgbm);
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wlsFilter =
        cv::ximgproc::createDisparityWLSFilter(sgbm);
    wlsFilter->setLambda(8000.0);
    wlsFilter->setSigmaColor(1.5);

    // -------------------------------------------------------------------------
    // NRF24L01 radio
    // -------------------------------------------------------------------------
    std::cout << "[INFO] Initialising NRF24L01 (CE="
              << static_cast<int>(RF24_CE_PIN)
              << " CSN=" << static_cast<int>(RF24_CSN_PIN) << ")...\n";

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    RF24 radio(RF24_CE_PIN, RF24_CSN_PIN);
    if (!radio.begin())
    {
        std::cerr << "[ERROR] NRF24L01 not responding - check SPI wiring.\n";
        return 1;
    }

    const uint8_t writePipe[] = "00002";
    const uint8_t readPipe[] = "00001";
    radio.openWritingPipe(writePipe);
    radio.openReadingPipe(1, readPipe);
    radio.setPALevel(RF24_PA_HIGH);
    radio.setDataRate(RF24_1MBPS);
    radio.stopListening();

    std::cout << "[INFO] Radio OK. Packet size = " << sizeof(SendDataStructure)
              << " bytes.\n";

    // -------------------------------------------------------------------------
    // State machine variables
    // -------------------------------------------------------------------------
    RobotState state = RobotState::WALKING_FORWARD;
    int16_t turnDirection = STICK_CENTER;

    auto stateStart = std::chrono::steady_clock::now();

    g_packet = neutralPacket();

    // Start dedicated radio transmit thread
    std::thread radioTx(radioThread, &radio);

    std::cout << "[INFO] Starting autonomous navigation. Ctrl+C to stop.\n";

    // =========================================================================
    // Main loop
    // =========================================================================
    while (g_running)
    {
        auto now = std::chrono::steady_clock::now();

        auto elapsedMs =
            [&](std::chrono::steady_clock::time_point since) -> long
        {
            return std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                                         since)
                .count();
        };

        // ---------------------------------------------------------------------
        // Capture stereo frames
        // ---------------------------------------------------------------------
        cv::Mat frameLeft, frameRight;
        if (!capLeft.read(frameLeft) || !capRight.read(frameRight))
        {
            std::cerr << "[WARN] Frame capture failed - retrying.\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        if (frameLeft.empty() || frameRight.empty())
        {
            std::cerr << "[WARN] Empty frame received - skipping.\n";
            continue;
        }

        // ---------------------------------------------------------------------
        // Stereo depth processing
        // ---------------------------------------------------------------------
        cv::Mat grayLeft, grayRight;
        cv::cvtColor(frameLeft, grayLeft, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frameRight, grayRight, cv::COLOR_BGR2GRAY);

        cv::Mat dispLeft, dispRight;
        sgbm->compute(grayLeft, grayRight, dispLeft);
        rightMatcher->compute(grayRight, grayLeft, dispRight);

        cv::Mat filteredDisp;
        wlsFilter->filter(dispLeft, grayLeft, filteredDisp, dispRight);

        cv::Mat dispFloat;
        filteredDisp.convertTo(dispFloat, CV_32F, 1.0 / 16.0);

        cv::Mat depthMap = disparityToDepth(dispFloat);
        DepthSample ds = sampleDepth(depthMap);

        // ---------------------------------------------------------------------
        // State machine
        // ---------------------------------------------------------------------
        SendDataStructure packet = neutralPacket();

        switch (state)
        {
        case RobotState::WALKING_FORWARD:
        {
            if (ds.center < OBSTACLE_DIST_M)
            {
                if (ds.left < ds.right)
                {
                    turnDirection = STICK_TURN_RIGHT;
                    std::cout << "[NAV] Obstacle at " << ds.center
                              << "m - turning RIGHT\n";
                }
                else
                {
                    turnDirection = STICK_TURN_LEFT;
                    std::cout << "[NAV] Obstacle at " << ds.center
                              << "m - turning LEFT\n";
                }
                stateStart = now;
                state = RobotState::AVOIDING_OBSTACLE;
                break;
            }

            // Ramp RFB from 512 down toward 0 over RAMP_DURATION_MS
            // RFB below centre = forward after robot brain applies * -1
            float rampFrac = static_cast<float>(elapsedMs(stateStart)) /
                             static_cast<float>(RAMP_DURATION_MS);
            if (rampFrac > 1.0f)
            {
                rampFrac = 1.0f;
            }

            int16_t currentRFB = static_cast<int16_t>(512 - rampFrac * 512);

            packet.RFB = currentRFB;
            packet.LT = STICK_CENTER;

            std::cout << "[NAV] Walking | RFB=" << currentRFB
                      << " ramp=" << static_cast<int>(rampFrac * 100) << "%"
                      << " depth=" << ds.center << "m\n";
            break;
        }

        case RobotState::AVOIDING_OBSTACLE:
        {
            if (elapsedMs(stateStart) >= TURN_DURATION_MS)
            {
                if (ds.center >= CLEAR_DIST_M)
                {
                    std::cout << "[NAV] Path clear at " << ds.center
                              << "m - resuming walk.\n";
                    stateStart = now;
                    state = RobotState::WALKING_FORWARD;
                }
                else
                {
                    std::cout << "[NAV] Still blocked at " << ds.center
                              << "m - continuing turn.\n";
                    stateStart = now;
                }
                break;
            }

            packet.RFB = STICK_CENTER;
            packet.LT = turnDirection;

            std::cout << "[NAV] Avoiding | LT=" << turnDirection
                      << " t=" << elapsedMs(stateStart) << "ms"
                      << " depth=" << ds.center << "m\n";
            break;
        }
        }

        // Update shared packet for radio thread
        {
            std::lock_guard<std::mutex> lock(g_packetMutex);
            g_packet = packet;
        }

    } // end main loop

    // =========================================================================
    // Shutdown
    // =========================================================================
    std::cout << "\n[INFO] Shutting down - sending stop burst.\n";
    {
        std::lock_guard<std::mutex> lock(g_packetMutex);
        g_packet = neutralPacket();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    radioTx.join();
    capLeft.release();
    capRight.release();

    std::cout << "[INFO] Stopped cleanly.\n";
    return 0;
}