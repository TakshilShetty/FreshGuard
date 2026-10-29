// ESP32 -> PC over USB serial. One line every 2 s:  gas_raw,temperature_c,humidity_percent
// Needs Arduino library: "DHT sensor library" by Adafruit (+ Adafruit Unified Sensor).
#include <DHT.h>
#define MQ_PIN  34        // MQ-3 analog out (AO) -> GPIO34. USE A VOLTAGE DIVIDER (ESP32 max 3.3 V)
#define DHT_PIN 4         // temperature/humidity data pin
#define DHT_TYPE DHT22    // change to DHT11 if that is your sensor
DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);          // 0..4095
  dht.begin();
  delay(2000);
  Serial.println("gas_raw,temperature_c,humidity_percent");
}

void loop() {
  long sum = 0;
  for (int i = 0; i < 16; i++) { sum += analogRead(MQ_PIN); delay(5); }
  int gas = sum / 16;                // average of 16 reads = less noise
  float t = dht.readTemperature(), h = dht.readHumidity();
  if (!isnan(t) && !isnan(h)) {      // never print fake numbers if the DHT fails
    Serial.print(gas); Serial.print(','); Serial.print(t, 1); Serial.print(','); Serial.println(h, 1);
  }
  delay(2000);
}
