/*
  ESP32 — Sign Language OLED Display (Fixed)
  ============================================
  Wiring:
    OLED VCC  → 3.3V
    OLED GND  → GND
    OLED SDA  → GPIO 21
    OLED SCL  → GPIO 22

  Libraries needed:
    - Adafruit SSD1306
    - Adafruit GFX Library
*/

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT  64
#define OLED_RESET     -1
#define OLED_ADDRESS  0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

String receivedText = "";

void setup() {
  // ✅ Must match Python script baud rate
  Serial.begin(115200);

  Wire.begin(21, 22);  // SDA=21, SCL=22

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
    Serial.println("OLED not found!");
    while (true);
  }

  // Startup screen
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("Sign Language");
  display.println("Detector");
  display.println("----------------");
  display.println("Waiting for");
  display.println("Python...");
  display.display();

  Serial.println("ESP32 Ready");
}

void loop() {
  // Read full line from Python
  if (Serial.available()) {
    char c = Serial.read();

    if (c == '\n') {
      // Got a complete message
      receivedText.trim();

      if (receivedText.length() > 0) {
        Serial.print("Got: ");
        Serial.println(receivedText);
        showGesture(receivedText);
        receivedText = "";
      }
    } else {
      receivedText += c;
    }
  }
}

void showGesture(String gesture) {
  display.clearDisplay();

  // ── Top bar ──────────────────────────────────────
  display.fillRect(0, 0, 128, 12, SSD1306_WHITE);
  display.setTextColor(SSD1306_BLACK);
  display.setTextSize(1);
  display.setCursor(2, 2);
  display.print("GESTURE:");

  // ── Big gesture label ─────────────────────────────
  display.setTextColor(SSD1306_WHITE);

  int len = gesture.length();

  if (len <= 5) {
    display.setTextSize(3);
    int x = (128 - len * 18) / 2;
    if (x < 0) x = 0;
    display.setCursor(x, 18);
  } else if (len <= 8) {
    display.setTextSize(2);
    int x = (128 - len * 12) / 2;
    if (x < 0) x = 0;
    display.setCursor(x, 20);
  } else {
    display.setTextSize(1);
    int x = (128 - len * 6) / 2;
    if (x < 0) x = 0;
    display.setCursor(x, 26);
  }

  display.println(gesture);

  // ── Bottom status bar ─────────────────────────────
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 56);
  display.print("Detecting...");

  display.display();
}
