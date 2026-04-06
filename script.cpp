/*
 * ============================================================
 *  Omi Eye Guard — ESP32-S3 Sense Firmware  v3.1
 *  Board  : Seeed XIAO ESP32S3 Sense
 *  Camera : OV2640 (worn as glasses, pointing at user's face)
 *
 *  PIPELINE:
 *    Capture 160×120 RGB565 → sub-sample to 80×60 grayscale →
 *    stream over BLE to phone app → app forwards to backend AI.
 *
 *  WHY RGB565 CAPTURE + GRAYSCALE OUTPUT:
 *    ESP32-S3's PIXFORMAT_GRAYSCALE has a DMA half-buffer bug that
 *    corrupts every 10th source row (period-5 banding after 2× sub-sample).
 *    RGB565 capture is clean; we convert to grayscale in startFrame().
 *    Output is still 4800 bytes / 27 chunks — same protocol as before.
 *
 *  BLE:
 *    Device: "Omi"
 *    Service UUID: 12340000-0000-0000-0000-000000000001
 *    FRAME char (NOTIFY): 12340000-0000-0000-0000-000000000004
 *      Packet: [seq:1][chunk:1][total:1][pixel data]
 *      Frame: 80×60 grayscale = 4800 bytes, 27 chunks × 179 bytes.
 *    CMD char (WRITE):    12340000-0000-0000-0000-000000000003
 *      0x01 = reserved
 *
 *  Arduino IDE Settings
 *  --------------------
 *  Board      : ESP32S3 Dev Module (or XIAO_ESP32S3)
 *  PSRAM      : OPI PSRAM
 *  USB CDC    : Enabled
 *  Flash      : 8 MB
 * ============================================================
 */

#include "esp_camera.h"
#include <string.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ============================================================
//  PIN MAP  (XIAO ESP32-S3 Sense)
// ============================================================
#define PWDN_GPIO_NUM   -1
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM   10
#define SIOD_GPIO_NUM   40
#define SIOC_GPIO_NUM   39

#define Y9_GPIO_NUM     48
#define Y8_GPIO_NUM     11
#define Y7_GPIO_NUM     12
#define Y6_GPIO_NUM     14
#define Y5_GPIO_NUM     16
#define Y4_GPIO_NUM     18
#define Y3_GPIO_NUM     17
#define Y2_GPIO_NUM     15

#define VSYNC_GPIO_NUM  38
#define HREF_GPIO_NUM   47
#define PCLK_GPIO_NUM   13

// ============================================================
//  FRAME
// ============================================================
#define FRAME_W          160
#define FRAME_H          120
#define FRAME_BYTES      (FRAME_W * FRAME_H * 2)   // 38400 (RGB565 capture)

// Sub-sampled grayscale frame sent over BLE
#define FRAME_SUB_W      80
#define FRAME_SUB_H      60
#define FRAME_SUB_PIXELS (FRAME_SUB_W * FRAME_SUB_H)   // 4800

// ============================================================
//  BLE
// ============================================================
#define DEVICE_NAME      "Omi"
#define SERVICE_UUID     "12340000-0000-0000-0000-000000000001"
#define CMD_CHAR_UUID    "12340000-0000-0000-0000-000000000003"
#define FRAME_CHAR_UUID  "12340000-0000-0000-0000-000000000004"

// BLE chunking — 27 chunks, same as the old grayscale firmware
#define FRAME_CHUNK_DATA    179
#define FRAME_TOTAL_CHUNKS  ((FRAME_SUB_PIXELS + FRAME_CHUNK_DATA - 1) / FRAME_CHUNK_DATA)

// ============================================================
//  TIMING
// ============================================================
#define LOOP_MS            50
#define FRAME_PERIOD_MS    2000
#define FRAME_CHUNKS_PER_LOOP  4

// ============================================================
//  GLOBALS
// ============================================================
BLEServer*          pServer    = nullptr;
BLECharacteristic*  pCmdChar   = nullptr;
BLECharacteristic*  pFrameChar = nullptr;

bool deviceConnected = false;
bool oldConnected    = false;
bool cameraReady     = false;

// 80×60 grayscale output buffer for BLE
static uint8_t  s_frameBuf[FRAME_SUB_PIXELS];

static bool     s_frameSending   = false;
static uint8_t  s_frameNextChunk = 0;
static uint8_t  s_frameSeq       = 0;

static uint32_t s_lastFrameMs = 0;

// ============================================================
//  BLE CALLBACKS
// ============================================================
class ServerCB : public BLEServerCallbacks {
  void onConnect(BLEServer*) override {
    deviceConnected = true;
    Serial.println("[BLE] Connected");
  }
  void onDisconnect(BLEServer*) override {
    deviceConnected = false;
    Serial.println("[BLE] Disconnected");
  }
};

class CmdCB : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* c) override {
    String v = c->getValue();
    if (v.length() > 0) {
      Serial.printf("[BLE] CMD: 0x%02X\n", (uint8_t)v[0]);
    }
  }
};

// ============================================================
//  CAMERA INIT — RGB565 (avoids ESP32-S3 grayscale DMA bug)
// ============================================================
bool initCamera() {
  camera_config_t cfg = {};

  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer   = LEDC_TIMER_0;

  cfg.pin_d0 = Y2_GPIO_NUM;   cfg.pin_d1 = Y3_GPIO_NUM;
  cfg.pin_d2 = Y4_GPIO_NUM;   cfg.pin_d3 = Y5_GPIO_NUM;
  cfg.pin_d4 = Y6_GPIO_NUM;   cfg.pin_d5 = Y7_GPIO_NUM;
  cfg.pin_d6 = Y8_GPIO_NUM;   cfg.pin_d7 = Y9_GPIO_NUM;

  cfg.pin_xclk  = XCLK_GPIO_NUM;
  cfg.pin_pclk  = PCLK_GPIO_NUM;
  cfg.pin_vsync = VSYNC_GPIO_NUM;
  cfg.pin_href  = HREF_GPIO_NUM;

  cfg.pin_sscb_sda = SIOD_GPIO_NUM;
  cfg.pin_sscb_scl = SIOC_GPIO_NUM;

  cfg.pin_pwdn  = PWDN_GPIO_NUM;
  cfg.pin_reset = RESET_GPIO_NUM;

  cfg.xclk_freq_hz = 20000000;           // 20 MHz (standard for OV2640)
  cfg.pixel_format = PIXFORMAT_RGB565;    // Clean on ESP32-S3 (no DMA Y-extraction bugs)
  cfg.frame_size   = FRAMESIZE_QQVGA;    // 160×120
  cfg.fb_count     = 1;
  cfg.grab_mode    = CAMERA_GRAB_LATEST;
  cfg.fb_location  = CAMERA_FB_IN_DRAM;

  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) {
    Serial.printf("[CAM] Init FAILED 0x%04x\n", err);
    return false;
  }

  sensor_t* s = esp_camera_sensor_get();
  s->set_exposure_ctrl(s, 1);
  s->set_aec2(s, 0);
  s->set_ae_level(s, 0);
  s->set_gain_ctrl(s, 1);
  s->set_gainceiling(s, (gainceiling_t)3);
  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_dcw(s, 1);
  s->set_bpc(s, 0);
  s->set_wpc(s, 1);
  s->set_raw_gma(s, 1);
  s->set_lenc(s, 0);
  s->set_brightness(s, 0);
  s->set_contrast(s, 0);
  s->set_saturation(s, 0);

  delay(1500);

  Serial.print("[CAM] Warm-up ");
  for (int i = 0; i < 12; i++) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
    Serial.print(".");
    delay(120);
  }
  Serial.println(" OK");
  return true;
}

// ============================================================
//  FRAME STREAMING — non-blocking
// ============================================================

// Convert one RGB565 pixel to 8-bit grayscale (BT.601 luminance).
static inline uint8_t rgb565_to_gray(const uint8_t* p) {
  uint16_t px = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
  uint8_t r = (px >> 11) & 0x1F;
  uint8_t g = (px >> 5)  & 0x3F;
  uint8_t b =  px        & 0x1F;
  return (uint8_t)(((r << 3) * 77 + (g << 2) * 150 + (b << 3) * 29) >> 8);
}

// Sub-sample 160×120 RGB565 → 80×60 grayscale with 2×2 AREA AVERAGING.
// Averaging 4 pixels per output pixel naturally denoises and anti-aliases
// instead of nearest-neighbour which throws away 75% of the sensor data.
static void startFrame(const uint8_t* buf) {
  for (int y = 0; y < FRAME_SUB_H; y++) {
    for (int x = 0; x < FRAME_SUB_W; x++) {
      const uint8_t* p = buf + ((y * 2) * FRAME_W + (x * 2)) * 2;
      uint16_t sum = (uint16_t)rgb565_to_gray(p)
                   + (uint16_t)rgb565_to_gray(p + 2)
                   + (uint16_t)rgb565_to_gray(p + FRAME_W * 2)
                   + (uint16_t)rgb565_to_gray(p + FRAME_W * 2 + 2);
      s_frameBuf[y * FRAME_SUB_W + x] = (uint8_t)(sum >> 2);
    }
  }
  s_frameSending   = true;
  s_frameNextChunk = 0;
}

static bool dripFrame() {
  if (!s_frameSending) return true;

  uint8_t total = (uint8_t)FRAME_TOTAL_CHUNKS;
  uint8_t packet[3 + FRAME_CHUNK_DATA];
  int sent = 0;

  while (s_frameNextChunk < total && sent < FRAME_CHUNKS_PER_LOOP) {
    uint32_t offset    = (uint32_t)s_frameNextChunk * FRAME_CHUNK_DATA;
    uint32_t remaining = FRAME_SUB_PIXELS - offset;
    uint32_t dataLen   = (remaining < FRAME_CHUNK_DATA) ? remaining : FRAME_CHUNK_DATA;

    packet[0] = s_frameSeq;
    packet[1] = s_frameNextChunk;
    packet[2] = total;
    memcpy(packet + 3, s_frameBuf + offset, dataLen);

    pFrameChar->setValue(packet, 3 + dataLen);
    pFrameChar->notify();

    s_frameNextChunk++;
    sent++;
  }

  if (s_frameNextChunk >= total) {
    Serial.printf("[BLE] Frame seq=%u SENT (%u chunks, %u bytes)\n",
                  (unsigned)s_frameSeq, (unsigned)total, (unsigned)FRAME_SUB_PIXELS);
    s_frameSeq++;
    s_frameSending = false;
    return true;
  }

  return false;
}

// ============================================================
//  BLE INIT
// ============================================================
void initBLE() {
  BLEDevice::init(DEVICE_NAME);
  BLEDevice::setMTU(512);

  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new ServerCB());

  BLEService* svc = pServer->createService(SERVICE_UUID);

  pCmdChar = svc->createCharacteristic(
    CMD_CHAR_UUID,
    BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR
  );
  pCmdChar->setCallbacks(new CmdCB());

  pFrameChar = svc->createCharacteristic(
    FRAME_CHAR_UUID,
    BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
  );
  pFrameChar->addDescriptor(new BLE2902());

  svc->start();

  BLEAdvertising* adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(SERVICE_UUID);
  adv->setScanResponse(true);
  adv->setMinPreferred(0x06);
  adv->setMaxPreferred(0x18);
  BLEDevice::startAdvertising();

  Serial.println("[BLE] Advertising as \"Omi\"");
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(800);

  Serial.println("========================================");
  Serial.println("  Omi Eye Guard v3.1 — RGB565→Gray      ");
  Serial.println("  Camera → BLE → App → Backend AI      ");
  Serial.println("========================================");

  cameraReady = initCamera();
  if (!cameraReady)
    Serial.println("[CAM] WARNING: camera failed — check PSRAM");

  initBLE();
  s_lastFrameMs = millis();
  Serial.println("[SYS] Ready");
}

// ============================================================
//  LOOP
// ============================================================
void loop() {
  delay(LOOP_MS);

  if (!deviceConnected && oldConnected) {
    delay(300);
    BLEDevice::startAdvertising();
    Serial.println("[BLE] Re-advertising");
    oldConnected = false;
  }
  if (deviceConnected && !oldConnected) oldConnected = true;

  if (!cameraReady || !deviceConnected) {
    if (!cameraReady) Serial.println("[WAIT] Camera not ready");
    if (!deviceConnected) Serial.println("[WAIT] No BLE connection");
    return;
  }

  if (s_frameSending) {
    dripFrame();
    return;
  }

  uint32_t now = millis();
  if (now - s_lastFrameMs < FRAME_PERIOD_MS) return;
  s_lastFrameMs = now;

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { Serial.println("[CAM] ERROR: Grab failed"); return; }
  if (fb->len < (size_t)FRAME_BYTES) {
    Serial.printf("[CAM] ERROR: Bad size %u (expected %u)\n", (unsigned)fb->len, (unsigned)FRAME_BYTES);
    esp_camera_fb_return(fb);
    return;
  }

  startFrame(fb->buf);
  esp_camera_fb_return(fb);

  Serial.printf("[CAM] Captured %ux%u RGB565, sub-sampled to %ux%u gray (%u bytes)\n",
                (unsigned)FRAME_W, (unsigned)FRAME_H,
                (unsigned)FRAME_SUB_W, (unsigned)FRAME_SUB_H,
                (unsigned)FRAME_SUB_PIXELS);
  Serial.printf("[BLE] Sending frame seq=%u as %u chunks (%u bytes)...\n",
                (unsigned)s_frameSeq, (unsigned)FRAME_TOTAL_CHUNKS, (unsigned)FRAME_SUB_PIXELS);
}
