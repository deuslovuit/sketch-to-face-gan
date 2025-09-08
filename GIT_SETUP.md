# 🚀 Git Setup Guide - อัปโหลดโปรเจ็คขึ้น GitHub

## 📋 ขั้นตอนการติดตั้งและอัปโหลด

### 1. 📥 ติดตั้ง Git

#### ทางเลือก A: ใช้ winget (แนะนำ)
```powershell
winget install --id Git.Git -e --source winget
```

#### ทางเลือก B: ดาวน์โหลดจากเว็บ
1. ไปที่ https://git-scm.com/download/win
2. ดาวน์โหลด Git for Windows
3. ติดตั้งตามขั้นตอน

### 2. 🔧 ตั้งค่า Git (ครั้งแรก)
```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 3. 📁 สร้าง .gitignore
สร้างไฟล์ .gitignore เพื่อไม่ให้อัปโหลดไฟล์ที่ไม่จำเป็น:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Data files (ขนาดใหญ่)
celebA/
*.png
*.jpg
*.jpeg

# Training output
training_output/
*.pth
*.log

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Temp files
*.tmp
*.temp
```

### 4. 🏠 สร้าง Local Repository
```powershell
cd d:\AI2
git init
git add .
git commit -m "Initial commit: Sketch-to-Face GAN project"
```

### 5. 🌐 สร้าง GitHub Repository

#### ทางเลือก A: ใช้ GitHub CLI (ถ้าติดตั้งแล้ว)
```powershell
gh repo create sketch-to-face-gan --public --description "AI project: Generate realistic faces from sketch images using Pix2Pix GAN"
git remote add origin https://github.com/yourusername/sketch-to-face-gan.git
git push -u origin main
```

#### ทางเลือก B: สร้างผ่านเว็บ GitHub
1. ไปที่ https://github.com/new
2. Repository name: `sketch-to-face-gan`
3. Description: `AI project: Generate realistic faces from sketch images using Pix2Pix GAN`
4. เลือก Public
5. กด "Create repository"

### 6. 🔗 เชื่อมต่อและอัปโหลด
```powershell
git remote add origin https://github.com/yourusername/sketch-to-face-gan.git
git branch -M main
git push -u origin main
```

## 📦 สิ่งที่จะอัปโหลด

### ✅ ไฟล์ที่จะอัปโหลด
- `train.py` - สคริปต์เทรน
- `trainer.py` - Training logic
- `models.py` - โครงสร้างโมเดล
- `dataset_simple.py` - DataLoader
- `data_preprocessing.py` - ประมวลผลข้อมูล
- `use_model.py` - ใช้โมเดล
- `requirements.txt` - Dependencies
- `README.md` - เอกสารโปรเจ็ค
- `cleanup_analysis.md` - วิเคราะห์ไฟล์

### ❌ ไฟล์ที่ไม่อัปโหลด (gitignore)
- `celebA/` - ข้อมูลต้นฉบับ (ขนาดใหญ่)
- `processed_data/` - ข้อมูลที่ประมวลผลแล้ว (ขนาดใหญ่)
- `training_output/` - ผลลัพธ์การเทรน (ขนาดใหญ่)
- `__pycache__/` - Python cache

## 🎯 ผลลัพธ์

หลังจากอัปโหลดเสร็จ คุณจะได้:
- 📂 Repository บน GitHub
- 🔗 Link สำหรับแชร์โปรเจ็ค
- 📋 เอกสารโปรเจ็คที่สมบูรณ์
- 🤝 พร้อมสำหรับ collaboration

## 💡 Tips

1. **ขนาดไฟล์**: ข้อมูลขนาดใหญ่ใช้ Git LFS หรือเก็บแยก
2. **Private repo**: เปลี่ยนเป็น Private ได้ในการตั้งค่า
3. **Branches**: ใช้ branches สำหรับ features ใหม่
4. **Issues**: ใช้ GitHub Issues ติดตามปัญหา

## 🚨 หมายเหตุ

- ข้อมูล CelebA และ processed_data ขนาดใหญ่ (หลายGB)
- แนะนำไม่อัปโหลดข้อมูลขึ้น GitHub
- ใช้เฉพาะโค้ดและเอกสาร
