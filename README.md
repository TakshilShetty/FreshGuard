# FreshGuard (SporeWatch) 🍌
> **AI-Assisted Banana Freshness Monitoring & Spoilage Prediction System**  
> *Engineering Lab (EL-SEM-3) Project*

FreshGuard is an end-to-end IoT and Machine Learning pipeline that continuously monitors banana ripening and spoilage dynamics in real time. It captures volatile gas emissions (ethanol/VOCs via MQ-3) alongside ambient temperature and relative humidity (DHT22) via an ESP32 microcontroller, streams records into MongoDB, and serves a live telemetry dashboard via Streamlit with mDNS network discovery.

---

## 📁 Project Structure

```text
EL-SEM-3/
├── app.py                     # Streamlit dashboard (real-time telemetry & UI)
├── start.py                   # Unified launcher (Streamlit + Caddy + mDNS + ESP32)
├── pyproject.toml             # Project packaging metadata & pytest configurations
├── requirements.txt           # Python package dependencies
├── .env.example               # Environment variable templates
├── README.md                  # System overview and quick-start instructions
├── assets/
│   └── spore_watch.css        # Custom UI styling and theme stylesheets
├── data/
│   ├── demo/                  # Synthetic multi-day test datasets (e.g. demo_5day_3bananas.csv)
│   ├── processed/             # Cleaned, standardized datasets for ML training
│   ├── raw/                   # Direct sensor captures from ESP32
│   └── README.md              # Data tiering and provenance architecture guide
├── firmware/
│   ├── esp32_sensor/
│   │   └── esp32_sensor.ino   # Arduino/ESP32 C++ firmware (MQ-3 + DHT22)
│   └── README.md              # Wiring schematic, libraries & flashing instructions
├── models/
│   ├── .gitkeep               # Directory for trained ML models (drop model.joblib here)
│   └── README.md              # Machine learning interface specifications
├── outputs/                   # Generated evaluation plots, exports, and metrics
├── scripts/                   # CLI utilities and background pipeline daemons
│   ├── collect_data.py        # Record calibrated sensor runs to data/raw/
│   ├── ingest.py              # Live ESP32 USB serial -> MongoDB streaming pipeline
│   ├── load_to_mongo.py       # Seed MongoDB with CSV data
│   ├── make_demo_data.py      # Generate synthetic 5-day decay simulation datasets
│   └── README.md              # CLI reference & usage examples
├── src/                       # Core application packages
│   ├── config.py              # Centralized environment & connection configurations
│   ├── data_manager.py        # Safe CSV persistence, atomic writes & origin tracking
│   ├── data_validator.py      # Schema enforcement & physical range validation
│   ├── predictor.py           # ML inference interface hook for model.joblib
│   ├── sensor_reader.py       # Serial line parser & dataframe builder
│   ├── ui_banana.py           # Dynamic SVG banana visualizer components
│   ├── ui_components.py       # Reusable Streamlit dashboard widgets & cards
│   └── ui_theme.py            # Design tokens, color palette, and system state
├── tests/                     # Test suite (114 pytest unit tests)
│   ├── test_data_manager.py   # Atomic file management and provenance tests
│   ├── test_data_validator.py # Range and anomaly validation tests
│   └── test_sensor_reader.py  # Serial parsing and dataframe creation tests
└── archive/                   # Archived historical zip releases (gitignored)
```

---

## 🔌 Hardware Setup

### Components
- **ESP32 Dev Module**
- **MQ-3 Gas Sensor** (sensitive to alcohol/ethanol emitted during fruit spoilage)
- **DHT22 Sensor** (temperature & relative humidity)
- Resistors for voltage divider (MQ-3 analog output is 5V; ESP32 ADC max is 3.3V)

### Pinout
| Sensor | Sensor Pin | ESP32 GPIO | Notes |
| :--- | :--- | :--- | :--- |
| **MQ-3** | VCC | 5V (VIN) | Needs ~24h pre-heat for steady baseline |
| **MQ-3** | GND | GND | Common ground |
| **MQ-3** | AO | GPIO34 | Via voltage divider to scale 0–5V -> 0–3.3V |
| **DHT22** | VCC | 3.3V / 5V | Depending on breakout board |
| **DHT22** | GND | GND | Common ground |
| **DHT22** | DATA | GPIO4 | Pull-up resistor (4.7kΩ–10kΩ) to VCC |

*See [`firmware/README.md`](firmware/README.md) for detailed schematics and library dependencies.*

---

## 🚀 Quick Start

### 1. Installation
Ensure Python 3.10+ is installed, then install the dependencies:
```bash
pip install -r requirements.txt
```

Ensure MongoDB is installed and running locally on default port `27017`:
```cmd
net start MongoDB
```

*(Optional: copy `.env.example` to `.env` to customize ports or connection URIs).*

### 2. Run the Full Stack with One Command
```bash
python start.py
```
This automatically:
1. Verifies local MongoDB connectivity on port `27017`.
2. Starts the Streamlit dashboard on port `5000`.
3. Starts a Caddy reverse proxy on port `80`.
4. Advertises the mDNS service on local Wi-Fi:
   - Local access: `http://localhost:5000` or `http://freshguard.local`
   - Mobile/LAN access: `http://<your-wifi-ip>`

If an ESP32 is plugged in over USB:
```bash
python start.py --port COM5 --specimen BANANA-LIVE
```

### 3. Standalone Dashboard (Demo / Offline Mode)
If you just want to run the dashboard without Caddy or mDNS:
```bash
streamlit run app.py
```
If MongoDB is not running, the dashboard automatically falls back to the deterministic synthetic dataset.

---

## 📊 Data Pipeline & CLI Scripts

- **Seed Demo Data to MongoDB**:
  ```bash
  python scripts/load_to_mongo.py data/demo/demo_5day_3bananas.csv
  ```
- **Record Real Data from ESP32 to `data/raw/`**:
  ```bash
  python scripts/collect_data.py --list-ports
  python scripts/collect_data.py --port COM5 --specimen BANANA-001 --minutes 30 --label ripe
  ```
- **Generate Fresh Synthetic Datasets**:
  ```bash
  python scripts/make_demo_data.py
  ```

*Backward-compatible root shortcuts (`python collect_data.py`, `python ingest.py`, etc.) are also preserved.*

---

## 🤖 Machine Learning Integration

Drop a trained scikit-learn model into `models/model.joblib`.
The predictor hook in [`src/predictor.py`](src/predictor.py) automatically detects it and runs live inference on incoming telemetry:
- Input features: `[gas_raw, temperature_c, humidity_percent]`
- Output: `{"label": "fresh"|"ripe"|"spoiled", "confidence": float}`

*See [`models/README.md`](models/README.md) for full specifications.*

---

## 🧪 Testing

Run the full automated test suite (114 unit tests):
```bash
pytest
```
All tests check data validation rules, file management atomicity, provenance checks, and serial parsing accuracy.
