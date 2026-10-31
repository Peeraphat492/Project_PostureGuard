# Posture Guard — ระบบติดตามท่านั่งแบบเรียลไทม์ด้วยกล้อง 2 มุม

Posture Guard เป็นแอปเดสก์ท็อป (Python + PyQt6) ที่ใช้กล้อง 2 ตัว คือกล้องด้านหน้าและกล้องด้านข้าง ร่วมกับ
MediaPipe Pose เพื่อตรวจท่านั่งของผู้ใช้คอมพิวเตอร์แบบเรียลไทม์ แจ้งเตือนเมื่อนั่งผิดท่าต่อเนื่อง
และชวนลุกขยับร่างกายด้วยเกมสั้น ๆ ที่ควบคุมด้วยท่าทาง

> เป็นเครื่องมือช่วยติดตามพฤติกรรมท่านั่ง ไม่ใช่เครื่องมือวินิจฉัยทางการแพทย์

## 1. ฟีเจอร์หลัก
- **ตรวจท่านั่งอัตโนมัติ** โดยไม่ต้อง calibrate: Forward Head, Neck Flexion, Slouching, Uneven Shoulder, Leaning Left/Right
  - กล้องหน้าวัดความเอียงของไหล่ ศีรษะ และลำตัว
  - กล้องข้างวัดศีรษะยื่นไปข้างหน้า การก้มคอ และการโน้มลำตัว
- **Posture Score 0–100** และสถานะ GOOD (80–100) / WARNING (60–79) / POOR (0–59) พร้อม skeleton overlay บนภาพกล้อง
- **ตรวจคุณภาพข้อมูล:** แจ้ง Partial Monitoring เมื่อใช้ได้เพียงกล้องเดียว และ Insufficient Data เมื่อมองไม่เห็นร่างกาย
  พร้อมคำแนะนำการวางกล้อง
- **Popup แจ้งเตือน** เมื่อท่าผิดต่อเนื่อง 10 วินาที
  - มีปุ่ม Snooze 1 min / Active Break / Dismiss
  - ปิดเองหลัง 15 วินาที หรือเมื่อกลับมานั่งถูกท่า 3 วินาที
  - เตือนให้พักเมื่อนั่งต่อเนื่องครบ 50 นาที
- **Active Break (2 นาที):** เล่นแบบ**ยืน**หน้ากล้องหน้า สลับ 2 เกม เกมละ 30 วินาที รวม 4 รอบ
  - **Stretch Game:** ทำท่ายืดเหยียดตามคำสั่งและค้างไว้ 1.5 วินาที
  - **Coin Game:** ใช้มือแตะเหรียญ และหลบระเบิดด้วยการย่อตัวหรือขยับตัวไปด้านข้าง
- **Session History:**
  - บันทึกอัตโนมัติทุก 15 วินาทีและเมื่อปิดแอป
  - ดูรายงานรายครั้ง กรองตามช่วงวัน (Today / 7 / 30 วัน) และดูกราฟแนวโน้มคะแนน
- **ข้อมูลอยู่ในเครื่องเท่านั้น:** เก็บใน SQLite เฉพาะตัวเลขสถิติ ไม่บันทึกภาพหรือวิดีโอ

## 2. เทคโนโลยีที่ใช้
Python 3.11 · PyQt6 6.6.1 · OpenCV 4.11 · MediaPipe Pose 0.10.21 · NumPy 1.26.4 · SQLite (มากับ Python)

## 3. ความต้องการของระบบ
- **ระบบปฏิบัติการ:** macOS หรือ Windows 10/11 (64-bit) พร้อม Python 3.11 (64-bit)
- **กล้อง 2 ตัวที่คอมพิวเตอร์มองเห็นพร้อมกัน** ซึ่งจำเป็นสำหรับการตรวจแบบเต็มรูปแบบ
  - **Front (กล้องหน้า):** วางตรงหน้าผู้ใช้ ให้เห็นศีรษะ ไหล่ และสะโพก
  - **Side (กล้องข้าง):** วางด้านข้างประมาณ 90° ให้เห็นหู ไหล่ และสะโพกฝั่งเดียวกัน ภาพต้องตั้งตรง ไม่ตะแคง
- **ทางเลือกกล้อง:**
  - **macOS:** กล้องในตัว + USB webcam หรือ iPhone ผ่าน Continuity Camera
  - **Windows:** เว็บแคมของเครื่อง + USB webcam หรือโทรศัพท์ผ่านแอปเว็บแคม เช่น DroidCam
    (Continuity Camera ใช้ได้เฉพาะ Mac)
- **สำหรับ Active Break:** ถอยห่างจากกล้องหน้าจนเห็นศีรษะ ไหล่ และแขนทั้งสองข้างขณะยืน

## 4. การติดตั้งและเปิดใช้งาน
เปิด terminal ในโฟลเดอร์โปรเจกต์ แล้วรันคำสั่งตามระบบปฏิบัติการ

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python camera_test.py
python main.py
```

- ถ้า PowerShell ไม่อนุญาตให้รัน `Activate.ps1` ให้ใช้ Command Prompt และรัน `.venv\Scripts\activate.bat` แทน
- บน Windows ต้องเปิดแอปด้วย `python main.py` เสมอ
  - `main.py` โหลด MediaPipe ก่อน PyQt6 ถ้าลำดับสลับกัน จะเกิด `DLL load failed while importing _framework_bindings`

**macOS (Terminal)**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python camera_test.py
python main.py
```

- ครั้งแรกที่เปิดกล้อง macOS จะขออนุญาตใช้กล้อง
  ให้อนุญาตแอปที่ใช้รัน (Terminal หรือ VS Code) ใน Privacy & Security › Camera

**คำสั่งเสริม**
- `python camera_test.py` ตรวจกล้องหมายเลข 0–5 และแสดงหมายเลขที่ใช้ได้
- `python self_test.py` ตรวจ logic หลักโดยไม่ใช้กล้อง

## 5. วิธีใช้งาน
1. เลือกหมายเลขกล้อง Front และ Side แล้วกด **Open Cameras** ระบบจะเริ่มตรวจท่านั่งทันที
2. ดูสถานะ คะแนน และปัญหาที่ตรวจพบบน Dashboard เมื่อนั่งผิดท่าต่อเนื่อง ระบบจะแสดง popup แจ้งเตือน
3. กด **Start Active Break** (หรือปุ่ม Active Break ใน popup) แล้วลุกขึ้นยืนเล่นเกมหน้ากล้องหน้า
   เมื่อครบ 2 นาที ระบบจะกลับมาตรวจท่านั่งเอง
4. กด **History** เพื่อดูประวัติและแนวโน้มคะแนน
5. **Stop Cameras** = พัก session (กด Open Cameras เพื่อทำต่อ) / ปิดแอป = จบ session และบันทึกผล

## 6. ภาพตัวอย่างการทำงาน
> **ภาพประกอบ (Demo):** ทุกภาพเป็นหน้าจอจริงของแอป แต่บุคคลในภาพกล้องเป็นหุ่นจำลอง (synthetic subject) และ
> ตำแหน่งข้อต่อ (landmarks) เป็นข้อมูลจำลองที่ป้อนแทนผลจาก MediaPipe **ภาพจึงไม่ใช่ผลการตรวจจับจริงของ MediaPipe**
> คะแนน การแจ้งเตือน และเกมในภาพเป็นผลที่แอปคำนวณจากข้อมูลจำลองนั้น ข้อมูลใน History เป็นข้อมูลตัวอย่าง

| | |
|---|---|
| ![Dashboard](docs/images/01_dashboard.png) **Dashboard** — นั่งถูกท่า | ![Dual camera monitoring](docs/images/02_dual_camera_monitoring.png) **Dual Camera Monitoring** — ตรวจพบ Forward Head + Uneven Shoulder |
| ![Posture warning popup](docs/images/03_posture_warning_popup.png) **Posture Warning** — popup หลังท่าผิด 10 วินาที | ![Session history](docs/images/06_session_history.png) **Session History** — สถิติและแนวโน้ม |
| ![Stretch game](docs/images/04_stretch_game.png) **Active Break: Stretch** — ยืนชูแขนทั้งสองข้าง | ![Coin game](docs/images/05_coin_game.png) **Active Break: Coin** — ยืนเตรียมหลบระเบิด |

## 7. ข้อควรทราบและข้อจำกัด
- **เกณฑ์และการวัด:**
  - เกณฑ์ท่าทางเป็นค่าทดลองที่อิงแนวปฏิบัติด้านการยศาสตร์ (OSHA, RULA) และยังไม่ผ่านการตรวจสอบทางคลินิก
  - วิเคราะห์จากภาพ 2 มิติ ความแม่นยำขึ้นกับตำแหน่งกล้อง แสง และการมองเห็นร่างกาย
- **สถิติและ History:**
  - นับเฉพาะช่วงที่กล้องทั้ง 2 ตัวให้ข้อมูลที่เชื่อถือได้
  - ถ้าใช้ได้เพียงกล้องเดียว (Partial Monitoring) แอปยังแสดงผลและแจ้งเตือน แต่ไม่นับเวลานั้นในสถิติและไม่บันทึกลง History
- **การทดสอบบน Windows:**
  - ทดสอบแล้วบน Windows 11 เครื่องเดียว (Python 3.11.9) โดยใช้เว็บแคมของเครื่องเป็นกล้องหน้า
    และ iPhone, iPad หรือ Android ผ่าน DroidCam เป็นกล้องข้าง
  - ยังไม่ได้ทดสอบบน Windows 10 และกับ USB webcam สองตัว
- **ข้อควรระวังเมื่อใช้ DroidCam:**
  - ต้องเชื่อมต่อแอปบนโทรศัพท์ก่อนกด Open Cameras
    มิฉะนั้นแอปจะแสดง "Both cameras live" แต่กล้องข้างจะไม่พบบุคคล
  - ถ้าภาพหมุนหรือตะแคง แอปจะทำงานแบบ Partial Monitoring จนกว่าจะตั้งภาพให้ตรง
- **ข้อจำกัดอื่น:**
  - การทดสอบส่วนใหญ่เป็นการทดสอบอัตโนมัติด้วยข้อมูลจำลอง การทดสอบกับผู้ใช้จริงยังมีจำกัด
  - การประมวลผลภาพจาก 2 กล้องพร้อมกันอาจช้าบนเครื่องสเปกต่ำ
- **ตำแหน่งไฟล์ History:**
  - macOS: `~/Library/Application Support/Posture Guard/history.sqlite3`
  - Windows: `%APPDATA%\Posture Guard\history.sqlite3`
