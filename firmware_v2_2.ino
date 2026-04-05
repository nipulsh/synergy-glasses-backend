/*
 * ============================================================
 *  Omi Eye Guard — ESP32-S3 Sense Firmware  v2.2  (COLOR)
 *  Board  : Seeed XIAO ESP32S3 Sense
 *  Camera : OV2640
 *
 *  PIPELINE:
 *    Capture 160x120 RGB565 -> sub-sample to 80x60 RGB565 ->
 *    stream over BLE -> app forwards to backend AI.
 *
 *  BLE Packet: [seq:1][chunk:1][total:1][pixel data]
 *  Frame: 80x60 RGB565 = 9600 bytes, 54 chunks x 179 bytes
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
//  FRAME — RGB565 = 2 bytes per pixel
// ============================================================
#define FRAME_W          160
#define FRAME_H          120
#define FRAME_BYTES      (FRAME_W * FRAME_H * 2)       // 38400

#define FRAME_SUB_W      80
#define FRAME_SUB_H      60
#define FRAME_SUB_BYTES  (FRAME_SUB_W * FRAME_SUB_H * 2)  // 9600

// ============================================================
//  BLE
// ============================================================
#define DEVICE_NAME      "Omi"
#define SERVICE_UUID     "12340000-0000-0000-0000-000000000001"
#define CMD_CHAR_UUID    "12340000-0000-0000-0000-000000000003"
#define FRAME_CHAR_UUID  "12340000-0000-0000-0000-000000000004"

#define FRAME_CHUNK_DATA    179
#define FRAME_TOTAL_CHUNKS  ((FRAME_SUB_BYTES + FRAME_CHUNK_DATA - 1) / FRAME_CHUNK_DATA)  // 54

// ============================================================
//  TIMING
// ============================================================
#define LOOP_MS                50
#define FRAME_PERIOD_MS        2000
#define FRAME_CHUNKS_PER_LOOP  4
#define BLE_STABILIZE_MS       450u

// ============================================================
//  GLOBALS
// ============================================================
BLEServer*          pServer    = nullptr;
BLECharacteristic*  pCmdChar   = nullptr;
BLECharacteristic*  pFrameChar = nullptr;

bool deviceConnected = false;
bool oldConnected    = false;
bool cameraReady     = false;

static uint32_t s_bleConnectMs  = 0;
static uint32_t s_lastFrameMs   = 0;
static uint32_t s_lastWaitLogMs = 0;

// Frame buffer — RGB565, 2 bytes/pixel (sub-sampled output for BLE)
static uint8_t s_frameBuf[FRAME_SUB_BYTES];      // 80x60 sub-sampled = 9600 bytes

// Chunk sender state
static bool    s_frameSending   = false;
static uint8_t s_frameNextChunk = 0;
static uint8_t s_frameSeq       = 0;

// ============================================================
//  BLE CALLBACKS
// ============================================================
class ServerCB : public BLEServerCallbacks {
  void onConnect(BLEServer*) override {
    deviceConnected = true;
    s_bleConnectMs  = millis();
    Serial.println("[BLE] Connected");
  }
  void onDisconnect(BLEServer*) override {
    deviceConnected  = false;
    s_frameSending   = false;
    s_frameNextChunk = 0;
    Serial.println("[BLE] Disconnected");
  }
};

class CmdCB : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* c) override {
    String v = c->getValue();
    if (v.length() > 0)
      Serial.printf("[BLE] CMD: 0x%02X\n", (uint8_t)v[0]);
  }
};

// ============================================================
//  CAMERA INIT
// ============================================================
bool initCamera() {
  camera_config_t cfg = {};

  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer   = LEDC_TIMER_0;

  cfg.pin_d0 = Y2_GPIO_NUM;  cfg.pin_d1 = Y3_GPIO_NUM;
  cfg.pin_d2 = Y4_GPIO_NUM;  cfg.pin_d3 = Y5_GPIO_NUM;
  cfg.pin_d4 = Y6_GPIO_NUM;  cfg.pin_d5 = Y7_GPIO_NUM;
  cfg.pin_d6 = Y8_GPIO_NUM;  cfg.pin_d7 = Y9_GPIO_NUM;

  cfg.pin_xclk     = XCLK_GPIO_NUM;
  cfg.pin_pclk     = PCLK_GPIO_NUM;
  cfg.pin_vsync    = VSYNC_GPIO_NUM;
  cfg.pin_href     = HREF_GPIO_NUM;
  cfg.pin_sscb_sda = SIOD_GPIO_NUM;
  cfg.pin_sscb_scl = SIOC_GPIO_NUM;
  cfg.pin_pwdn     = PWDN_GPIO_NUM;
  cfg.pin_reset    = RESET_GPIO_NUM;

  cfg.xclk_freq_hz = 20000000;          // OV2640 needs 20 MHz for RGB565
  cfg.pixel_format = PIXFORMAT_RGB565;  // COLOR — 2 bytes/pixel
  cfg.frame_size   = FRAMESIZE_QQVGA;   // 160x120
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
  s->set_aec2(s, 1);
  s->set_ae_level(s, 0);
  s->set_gain_ctrl(s, 1);
  s->set_gainceiling(s, (gainceiling_t)6);
  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_dcw(s, 1);
  s->set_bpc(s, 0);
  s->set_wpc(s, 1);
  s->set_raw_gma(s, 1);
  s->set_lenc(s, 1);
  s->set_brightness(s, 1);
  s->set_contrast(s, 1);
  s->set_saturation(s, 0);

  delay(500);

  Serial.print("[CAM] Warm-up ");
  for (int i = 0; i < 20; i++) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
    Serial.print(".");
    delay(80);
  }
  Serial.println(" OK");
  return true;
}

// ============================================================
//  FRAME STREAMING
// ============================================================

// Sub-sample 160x120 RGB565 -> 80x60 RGB565 (nearest neighbour, 2x2 skip).
// src_row_bytes = bytes per camera line (fb->len / fb->height when DMA pads rows).
static void startFrame(const uint8_t* buf, size_t src_row_bytes) {
  for (int y = 0; y < FRAME_SUB_H; y++) {
    for (int x = 0; x < FRAME_SUB_W; x++) {
      size_t src = (size_t)(y * 2) * src_row_bytes + (size_t)(x * 2) * 2;
      int dst = (y * FRAME_SUB_W + x) * 2;
      s_frameBuf[dst]     = buf[src];
      s_frameBuf[dst + 1] = buf[src + 1];
    }
  }
  s_frameSending   = true;
  s_frameNextChunk = 0;
}

static bool dripFrame() {
  if (!s_frameSending) return true;
  if (!deviceConnected || pFrameChar == nullptr) {
    s_frameSending = false;
    return true;
  }

  uint8_t total = (uint8_t)FRAME_TOTAL_CHUNKS;
  uint8_t packet[3 + FRAME_CHUNK_DATA];
  int sent = 0;

  while (s_frameNextChunk < total && sent < FRAME_CHUNKS_PER_LOOP) {
    uint32_t offset    = (uint32_t)s_frameNextChunk * FRAME_CHUNK_DATA;
    uint32_t remaining = FRAME_SUB_BYTES - offset;
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
    Serial.printf("[BLE] Frame seq=%u SENT (%u chunks, %u bytes RGB565)\n",
                  (unsigned)s_frameSeq, (unsigned)total, (unsigned)FRAME_SUB_BYTES);
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
  BLEDevice::setMTU(517);

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
  Serial.println("  Omi Eye Guard v2.2 — RGB565 Color    ");
  Serial.println("  Camera -> BLE -> App -> Backend AI   ");
  Serial.println("========================================");
  Serial.printf("[INFO] %dx%d -> %dx%d RGB565, %u bytes, %u chunks\n",
                FRAME_W, FRAME_H, FRAME_SUB_W, FRAME_SUB_H,
                (unsigned)FRAME_SUB_BYTES, (unsigned)FRAME_TOTAL_CHUNKS);

  cameraReady = initCamera();
  if (!cameraReady)
    Serial.println("[CAM] WARNING: camera failed");

  initBLE();
  s_lastFrameMs = millis();
  Serial.println("[SYS] Ready");
}

// ============================================================
//  LOOP
// ============================================================
void loop() {
  delay(LOOP_MS);

  // BLE reconnection
  if (!deviceConnected && oldConnected) {
    delay(300);
    BLEDevice::startAdvertising();
    Serial.println("[BLE] Re-advertising");
    oldConnected = false;
  }
  if (deviceConnected && !oldConnected) oldConnected = true;

  if (!cameraReady || !deviceConnected) {
    uint32_t nw = millis();
    if ((uint32_t)(nw - s_lastWaitLogMs) >= 2000u) {
      s_lastWaitLogMs = nw;
      if (!cameraReady)     Serial.println("[WAIT] Camera not ready");
      if (!deviceConnected) Serial.println("[WAIT] No BLE connection");
    }
    return;
  }

  // Wait for MTU + CCCD handshake after connect
  if ((uint32_t)(millis() - s_bleConnectMs) < BLE_STABILIZE_MS) return;

  // Keep dripping chunks if mid-frame
  if (s_frameSending) {
    dripFrame();
    return;
  }

  // Time for a new frame?
  uint32_t now = millis();
  if (now - s_lastFrameMs < FRAME_PERIOD_MS) return;
  s_lastFrameMs = now;

  // Capture
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { Serial.println("[CAM] ERROR: Grab failed"); return; }

  if (fb->len < (size_t)FRAME_BYTES || fb->height == 0) {
    Serial.printf("[CAM] ERROR: Bad fb len=%u h=%u (need >= %u)\n",
                  (unsigned)fb->len, (unsigned)fb->height, (unsigned)FRAME_BYTES);
    esp_camera_fb_return(fb);
    return;
  }

  size_t src_row = fb->len / fb->height;
  if (fb->len % fb->height != 0 || src_row < (size_t)FRAME_W * 2u) {
    Serial.printf("[CAM] WARN: odd fb layout len=%u h=%u row=%u, using %u\n",
                  (unsigned)fb->len, (unsigned)fb->height, (unsigned)src_row,
                  (unsigned)(FRAME_W * 2));
    src_row = (size_t)FRAME_W * 2u;
  }

  startFrame(fb->buf, src_row);
  esp_camera_fb_return(fb);

  Serial.printf("[CAM] Captured %dx%d RGB565 (row=%u bytes)\n",
                FRAME_W, FRAME_H, (unsigned)src_row);

  // BLE send (startFrame already armed)
  Serial.printf("[BLE] Sending frame seq=%u as %u chunks (%u bytes RGB565)...\n",
                (unsigned)s_frameSeq, (unsigned)FRAME_TOTAL_CHUNKS, (unsigned)FRAME_SUB_BYTES);
}
