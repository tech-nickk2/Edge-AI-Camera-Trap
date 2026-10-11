// Edge-AI camera trap: sensor side (STM32 microcontroller of the UNO Q).
//
// Watches the PIR sensor. On motion it reads the LDR, switches the IR LEDs on
// if it's dark, and tells the Linux side (python/main.py) through the Bridge.
// The Linux side calls "ir_off" when it has finished capturing; a timeout
// switches the LEDs off anyway in case that call never arrives.
//
// Wiring:
//   PIR  (e.g. HC-SR501): VCC -> 5V, GND -> GND, OUT -> D2
//   LDR: one leg -> 3.3V, other leg -> A0; 10k resistor from A0 -> GND
//        (reading goes UP as the light gets brighter)
//   IR LEDs: switched by a logic-level N-channel MOSFET
//        gate -> D3 through 100R, 10k from gate -> GND,
//        LEDs (with their resistors) from 5V -> drain, source -> GND

#include <Arduino_RouterBridge.h>

const int PIR_PIN = 2;
const int LDR_PIN = A0;
const int IR_PIN  = 3;

// Calibrate: the Linux side logs the light value with every trigger.
// Night mode starts below DARK_BELOW and ends above LIGHT_ABOVE (the gap
// stops it flickering between modes at dusk).
const int DARK_BELOW  = 300;
const int LIGHT_ABOVE = 400;

const unsigned long COOLDOWN_MS  = 5000;   // ignore new triggers for this long
const unsigned long IR_MAX_ON_MS = 15000;  // safety timeout for the IR LEDs

bool night = false;
bool lastPir = false;
bool irOn = false;
unsigned long lastTrigger = 0;
unsigned long irOnSince = 0;

void setIr(bool on) {
  digitalWrite(IR_PIN, on ? HIGH : LOW);
  irOn = on;
  if (on) irOnSince = millis();
}

void irOff() {
  setIr(false);
}

void setup() {
  pinMode(PIR_PIN, INPUT);
  pinMode(IR_PIN, OUTPUT);
  setIr(false);

  Bridge.begin();
  Bridge.provide("ir_off", irOff);
}

void loop() {
  int light = analogRead(LDR_PIN);
  if (night && light > LIGHT_ABOVE) night = false;
  else if (!night && light < DARK_BELOW) night = true;

  bool pir = digitalRead(PIR_PIN) == HIGH;
  if (pir && !lastPir && millis() - lastTrigger > COOLDOWN_MS) {
    lastTrigger = millis();
    if (night) setIr(true);  // light the scene before the Linux side grabs frames
    Bridge.notify("motion", night, light);
  }
  lastPir = pir;

  if (irOn && millis() - irOnSince > IR_MAX_ON_MS) setIr(false);
  delay(20);
}
