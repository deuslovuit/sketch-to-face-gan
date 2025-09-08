# 🎨 Sketch-to-Face GAN Project
## โปรเจ็คสร้างใบหน้าจริงจากภาพสเก็ต

### 📁 โครงสร้างโปรเจ็ค (หลังทำความสะอาด)

```
d:\AI2/
├── 🤖 Core Files
│   ├── train.py              # สคริปต์หลักสำหรับเทรนโมเดล
│   ├── trainer.py            # Class จัดการการเทรน Pix2Pix
│   ├── models.py             # Generator (U-Net) + Discriminator
│   ├── dataset_simple.py     # DataLoader สำหรับ sketch-image pairs
│   ├── data_preprocessing.py # ประมวลผล CelebA เป็น sketches
│   └── use_model.py          # ใช้โมเดลที่เทรนแล้ว
│
├── 📊 Data
│   ├── celebA/               # ข้อมูลต้นฉบับ (30,000 ภาพ)
│   ├── processed_data/       # ข้อมูลประมวลผลแล้ว
│   │   ├── images/           # ภาพหน้าจริง (29,999 ไฟล์)
│   │   └── sketches/         # ภาพสเก็ต (29,999 ไฟล์)
│   └── training_output/      # ผลลัพธ์การเทรน
│       ├── checkpoints/      # โมเดลที่บันทึก
│       ├── logs/             # Training logs
│       └── samples/          # ภาพตัวอย่างระหว่างเทรน
│
├── ⚙️ Config
│   ├── requirements.txt      # Dependencies
│   └── cleanup_analysis.md   # การวิเคราะห์ไฟล์
│
└── 🗂️ Cache
    └── __pycache__/          # Python cache
```

### 🎯 วิธีใช้งาน

#### 1. เทรนโมเดล
```bash
python train.py
```

#### 2. ใช้โมเดลที่เทรนแล้ว
```bash
python use_model.py
```

#### 3. ประมวลผลข้อมูลใหม่ (ถ้าต้องการ)
```bash
python data_preprocessing.py
```

### 📊 ข้อมูลโปรเจ็ค

- **Dataset**: CelebA (29,999 ภาพใบหน้า)
- **Model**: Pix2Pix GAN (Generator + Discriminator)
- **Input**: Sketch images (256x256)
- **Output**: Realistic face images (256x256)
- **Framework**: PyTorch 2.8.0
- **Training**: 50 epochs, 5,000 samples

### 🧹 ไฟล์ที่ลบไปแล้ว

ลบไฟล์ที่ไม่จำเป็นออกไป 19 ไฟล์:
- ไฟล์ทดสอบ (7 ไฟล์)
- ไฟล์ setup GPU (5 ไฟล์) 
- ไฟล์สรุป/วิเคราะห์ (3 ไฟล์)
- ไฟล์ซ้ำซ้อน (1 ไฟล์)
- ไฟล์ภาพผลลัพธ์ (3 ไฟล์)

### ✅ ผลลัพธ์

โปรเจ็คตอนนี้:
- ✨ สะอาดและเข้าใจง่าย
- 🎯 มีเฉพาะไฟล์ที่จำเป็น
- 🚀 พร้อมสำหรับการใช้งานและพัฒนาต่อ
- 📦 ขนาดเล็กลง (ประหยัด ~1MB)
