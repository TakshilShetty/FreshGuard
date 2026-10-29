/*
 * FreshGuard ESP32 Multi-Sensor Transit & Spoilage Firmware
 * =========================================================
 * Universal Driver supporting:
 *   - Genuine InvenSense MPU-6050
 *   - Clone / Counterfeit / Re-marked IMUs (Hangshun, Geehy, Silan, MPU-6500)
 *   - Automatic I2C address detection (0x68 and 0x69)
 *   - Automatic Raw Register Fallback if Adafruit WHO_AM_I check fails
 *
 * Hardware Configuration:
 *   - MQ-3 Gas Sensor (Analog ADC: GPIO 34 via voltage divider)
 *   - DHT22 Temp & Humidity (Digital: GPIO 4)
 *   - MPU-6050 / Clone IMU (I2C: SDA=GPIO 21, SCL=GPIO 22)
 *   - Warning LED (GPIO 2)
 *   - Alert Buzzer (GPIO 13)
 *
 * CSV Output Format (every 2 seconds):
 *   gas_raw,temperature_c,humidity_percent,accel_x,accel_y,accel_z,shock_g,pitch_deg,roll_deg
 */

#include <Wire.h>
#include <DHT.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#define MQ_PIN        34      // MQ-3 analog pin (GPIO34 is input-only ADC1)
#define DHT_PIN       4       // DHT22 data pin
#define DHT_TYPE      DHT22   // or DHT11
#define LED_PIN       2       // Onboard warning LED
#define BUZZER_PIN    13      // Actuator/Buzzer pin

#define MPU_REG_PWR_MGMT_1   0x6B
#define MPU_REG_CONFIG       0x1A
#define MPU_REG_ACCEL_CONFIG 0x1C
#define MPU_REG_ACCEL_XOUT_H 0x3B
#define MPU_REG_WHO_AM_I     0x75

enum ImuDriverMode {
  IMU_MODE_NONE,
  IMU_MODE_ADAFRUIT,
  IMU_MODE_RAW_I2C
};

DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_MPU6050 mpu;

bool mpu_available = false;
ImuDriverMode imu_mode = IMU_MODE_NONE;
uint8_t mpu_address = 0x68;
uint8_t mpu_who_am_i = 0x00;

// Sampling & Telemetry intervals
unsigned long last_transmit_ms = 0;
const unsigned long TRANSMIT_INTERVAL_MS = 2000; // Emit telemetry every 2 seconds

// IMU Shock and Orientation metrics
float peak_shock_g = 1.0;
float live_ax = 0.0, live_ay = 0.0, live_az = 1.0;
float live_pitch = 0.0, live_roll = 0.0;

// Helper: Read a single register over I2C
uint8_t i2c_read_reg(uint8_t addr, uint8_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0xFF;
  if (Wire.requestFrom((uint8_t)addr, (size_t)1) == 1) {
    return Wire.read();
  }
  return 0xFF;
}

// Helper: Write a single register over I2C
bool i2c_write_reg(uint8_t addr, uint8_t reg, uint8_t data) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(data);
  return (Wire.endTransmission() == 0);
}

// Universal IMU Initializer: Supports genuine chips and all clone variants
void init_imu_universal() {
  // 1. Detect I2C Address (0x68 or 0x69)
  mpu_address = 0;
  Wire.beginTransmission(0x68);
  if (Wire.endTransmission() == 0) {
    mpu_address = 0x68;
  } else {
    Wire.beginTransmission(0x69);
    if (Wire.endTransmission() == 0) {
      mpu_address = 0x69;
    }
  }

  if (mpu_address == 0) {
    Serial.println("# [IMU WARNING] No IMU found at 0x68 or 0x69. Check VCC/GND/SDA=21/SCL=22.");
    mpu_available = false;
    imu_mode = IMU_MODE_NONE;
    return;
  }

  // 2. Read WHO_AM_I register (0x75)
  mpu_who_am_i = i2c_read_reg(mpu_address, MPU_REG_WHO_AM_I);
  Serial.print("# [IMU INFO] Found I2C device at 0x");
  Serial.print(mpu_address, HEX);
  Serial.print(" | WHO_AM_I = 0x");
  Serial.println(mpu_who_am_i, HEX);

  // 3. Try official Adafruit library first
  if (mpu.begin(mpu_address)) {
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    mpu_available = true;
    imu_mode = IMU_MODE_ADAFRUIT;
    Serial.println("# [IMU STATUS] Initialized via Adafruit MPU-6050 driver (Genuine/Full Compatible).");
    return;
  }

  // 4. Fallback: Clone / Re-marked Chip Hardware Bypass
  // Clones fail Adafruit begin() because WHO_AM_I is 0x70, 0x72, 0x98, 0x12, etc.
  // We wake up the chip and configure registers directly over I2C.
  Serial.println("# [IMU STATUS] Adafruit ID check failed -> Activating Universal Clone Direct-I2C driver.");
  
  // Wake up chip (clear SLEEP bit in PWR_MGMT_1)
  i2c_write_reg(mpu_address, MPU_REG_PWR_MGMT_1, 0x00);
  delay(10);
  
  // Configure ±8g range (0x10) -> Sensitivity: 4096 LSB/g
  i2c_write_reg(mpu_address, MPU_REG_ACCEL_CONFIG, 0x10);
  
  // Configure Digital Low Pass Filter ~42Hz (0x03)
  i2c_write_reg(mpu_address, MPU_REG_CONFIG, 0x03);
  delay(10);

  // Verify we can read 6 accel bytes
  Wire.beginTransmission(mpu_address);
  Wire.write(MPU_REG_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) == 0 && Wire.requestFrom((uint8_t)mpu_address, (size_t)6) == 6) {
    mpu_available = true;
    imu_mode = IMU_MODE_RAW_I2C;
    Serial.println("# [IMU STATUS] Clone / Alternative IMU successfully activated! Full transit tracking active.");
  } else {
    Serial.println("# [IMU ERROR] Clone initialization failed to read accel registers.");
    mpu_available = false;
    imu_mode = IMU_MODE_NONE;
  }
}

// Read acceleration from IMU (supporting both Adafruit and Clone Raw modes)
bool read_imu_acceleration(float &ax, float &ay, float &az) {
  if (!mpu_available) return false;

  if (imu_mode == IMU_MODE_ADAFRUIT) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    ax = a.acceleration.x / 9.80665f;
    ay = a.acceleration.y / 9.80665f;
    az = a.acceleration.z / 9.80665f;
    return true;
  } else if (imu_mode == IMU_MODE_RAW_I2C) {
    Wire.beginTransmission(mpu_address);
    Wire.write(MPU_REG_ACCEL_XOUT_H);
    if (Wire.endTransmission(false) == 0 && Wire.requestFrom((uint8_t)mpu_address, (size_t)6) == 6) {
      int16_t raw_x = (Wire.read() << 8) | Wire.read();
      int16_t raw_y = (Wire.read() << 8) | Wire.read();
      int16_t raw_z = (Wire.read() << 8) | Wire.read();

      // ±8g sensitivity: 4096 LSB/g
      ax = (float)raw_x / 4096.0f;
      ay = (float)raw_y / 4096.0f;
      az = (float)raw_z / 4096.0f;
      return true;
    }
  }
  return false;
}

void print_diagnostic_info() {
  Serial.println("\n# ===== FRESHGUARD HARDWARE DIAGNOSTIC =====");
  Serial.print("# MQ-3 Raw ADC: ");
  Serial.println(analogRead(MQ_PIN));
  Serial.print("# DHT22 Temp: ");
  Serial.print(dht.readTemperature(), 1);
  Serial.print(" C | Hum: ");
  Serial.print(dht.readHumidity(), 1);
  Serial.println(" %");
  Serial.print("# IMU Address: 0x");
  Serial.print(mpu_address, HEX);
  Serial.print(" | WHO_AM_I: 0x");
  Serial.println(mpu_who_am_i, HEX);
  Serial.print("# IMU Mode: ");
  if (imu_mode == IMU_MODE_ADAFRUIT) Serial.println("Adafruit Driver (Genuine/Compatible)");
  else if (imu_mode == IMU_MODE_RAW_I2C) Serial.println("Universal Clone Driver (Direct Register I2C)");
  else Serial.println("NOT DETECTED");
  Serial.println("# ==========================================\n");
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12); // ESP32 12-bit ADC (0 - 4095)

  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  dht.begin();
  Wire.begin(21, 22); // I2C pins: SDA=21, SCL=22
  Wire.setClock(400000); // 400kHz Fast Mode I2C

  // Initialize IMU with universal clone-proof fallback
  init_imu_universal();

  delay(1000);
  print_diagnostic_info();

  // CSV Telemetry Header
  Serial.println("gas_raw,temperature_c,humidity_percent,accel_x,accel_y,accel_z,shock_g,pitch_deg,roll_deg");
}

void process_serial_commands() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "ALERT:ON") {
      digitalWrite(LED_PIN, HIGH);
      digitalWrite(BUZZER_PIN, HIGH);
    } else if (cmd == "ALERT:OFF") {
      digitalWrite(LED_PIN, LOW);
      digitalWrite(BUZZER_PIN, LOW);
    } else if (cmd == "RESET_TRIP") {
      peak_shock_g = 1.0;
    } else if (cmd == "HW:CHECK" || cmd == "DIAG") {
      print_diagnostic_info();
    }
  }
}

void loop() {
  process_serial_commands();

  // High-frequency IMU loop (~50Hz) to capture instantaneous impacts & drops
  if (mpu_available) {
    float cur_x, cur_y, cur_z;
    if (read_imu_acceleration(cur_x, cur_y, cur_z)) {
      live_ax = cur_x;
      live_ay = cur_y;
      live_az = cur_z;

      // Instantaneous total acceleration vector magnitude
      float current_shock = sqrt(live_ax * live_ax + live_ay * live_ay + live_az * live_az);
      if (current_shock > peak_shock_g) {
        peak_shock_g = current_shock;
      }

      // Pitch and Roll calculation (in degrees)
      live_pitch = atan2(-live_ax, sqrt(live_ay * live_ay + live_az * live_az)) * 180.0f / PI;
      live_roll  = atan2(live_ay, live_az) * 180.0f / PI;
    }
  }

  // Periodic Telemetry Transmission
  unsigned long now = millis();
  if (now - last_transmit_ms >= TRANSMIT_INTERVAL_MS) {
    last_transmit_ms = now;

    // 16-sample oversampling for MQ-3 analog noise reduction
    long gas_sum = 0;
    for (int i = 0; i < 16; i++) {
      gas_sum += analogRead(MQ_PIN);
      delay(2);
    }
    int gas_raw = gas_sum / 16;

    // Read DHT22
    float t = dht.readTemperature();
    float h = dht.readHumidity();

    if (!isnan(t) && !isnan(h)) {
      // Format: gas_raw,temperature_c,humidity_percent,accel_x,accel_y,accel_z,shock_g,pitch_deg,roll_deg
      Serial.print(gas_raw);
      Serial.print(",");
      Serial.print(t, 1);
      Serial.print(",");
      Serial.print(h, 1);
      Serial.print(",");
      Serial.print(live_ax, 2);
      Serial.print(",");
      Serial.print(live_ay, 2);
      Serial.print(",");
      Serial.print(live_az, 2);
      Serial.print(",");
      Serial.print(peak_shock_g, 2);
      Serial.print(",");
      Serial.print(live_pitch, 1);
      Serial.print(",");
      Serial.println(live_roll, 1);
    }

    // Decay peak shock towards 1.0g for next interval
    peak_shock_g = 1.0;
  }

  delay(20); // 50Hz IMU polling cycle
}
