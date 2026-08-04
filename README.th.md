# pypdpg

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/getting_started.ipynb)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: PDPG Community](https://img.shields.io/badge/license-PDPG%20Community-blueviolet)

รันงานประมวลผลข้อมูลด้วย Python บนข้อมูลที่เข้ารหัสแบบโฮโมมอร์ฟิก
(homomorphic encryption)
**ผู้ควบคุมข้อมูลส่วนบุคคล** (data controller) เข้ารหัสข้อมูลและเก็บกุญแจลับไว้กับตัว
**ผู้ประมวลผลข้อมูลส่วนบุคคล** (data processor) คำนวณบนข้อมูลเข้ารหัสด้วยโค้ดเดิม
โดยไม่มีทางอ่านข้อมูลได้ แล้วผู้ควบคุมข้อมูลเป็นผู้ถอดรหัสผลลัพธ์

pypdpg กำลังขยายการรองรับไลบรารีที่ใช้กันทั่วไปในงานข้อมูล Python ทีละตัว —
มี numpy เป็นแกนกลาง ต่อยอดด้วยเลเยอร์ pandas และ scikit-learn
การรองรับยังเป็นบางส่วนและเพิ่มขึ้นเรื่อย ๆ ส่วนที่ทำไม่ได้ระบบจะปฏิเสธพร้อมคำอธิบาย
— โครงการโดย [PDPG-lab](https://pdpglab.xyz)

ตัวเข้ารหัสเบื้องหลังเป็น "แบ็กเอนด์" ที่สลับได้: ปัจจุบันใช้ TenSEAL (CKKS)
เป็นแบ็กเอนด์ตัวแรก และมีแผนรองรับแบ็กเอนด์ตระกูล TFHE
(เปรียบเทียบค่าได้แบบแม่นยำ ความลึกไม่จำกัด) ในอนาคต —
เมื่อแบ็กเอนด์ใหม่มาถึง โค้ดฝั่งผู้ใช้ไม่ต้องแก้ แบ็กเอนด์เป็นหน้าที่ของเรา

## ทำไมจึงสำคัญ

ภายใต้ พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562 (PDPA) และ GDPR
การส่งข้อมูลส่วนบุคคลให้ผู้ประมวลผลภายนอกเป็นเรื่องที่ต้องมีมาตรการรองรับ
pypdpg ทำให้ส่ง *เฉพาะข้อมูลที่เข้ารหัสแล้ว* ได้ โดยที่ฝั่งผู้ประมวลผล
แทบไม่ต้องแก้โค้ดเดิมเลย: เข้ารหัสไป → เข้ารหัสกลับ →
ถอดรหัสได้เฉพาะเจ้าของกุญแจ

## การติดตั้ง

```
pip install git+https://github.com/PDPG-lab/pypdpg
```

## ตัวอย่างการใช้งาน

```python
import pypdpg as pdpg

# ผู้ควบคุมข้อมูล: สร้างกุญแจ เข้ารหัส แล้วส่งไฟล์
ctx = pdpg.Context.create()
ctx.save("controller.key")           # มีกุญแจลับ — เก็บไว้กับผู้ควบคุมข้อมูลเท่านั้น
ctx.save_public("processor.ctx")     # กุญแจสำหรับคำนวณ — ส่งให้ผู้ประมวลผลได้
pdpg.encrypt(X, ctx).save("data.enc")

# ผู้ประมวลผล: ไม่มีกุญแจลับในเครื่อง
pdpg.activate("processor.ctx")
pdpg.install()                       # ทำให้ np.load อ่านไฟล์ .enc ได้
X = np.load("data.enc")              # CipherArray — อ่านค่าไม่ได้ คำนวณได้
scores = X @ w + b                   # โค้ด numpy เดิม ไม่ต้องแก้
scores.save("result.enc")

# ผู้ควบคุมข้อมูล: ถอดรหัสผลลัพธ์
pdpg.activate("controller.key")
result = pdpg.load("result.enc").decrypt()
```

รองรับ DataFrame ของ pandas (คงชื่อคอลัมน์ไว้ในไฟล์เข้ารหัส), โมเดลเชิงเส้นของ
scikit-learn ที่ฝึกเสร็จแล้ว (`pdpg.sklearn.wrap` — ฝึกบนข้อมูลจริงฝั่งเจ้าของโมเดล
แล้วนำมาคำนวณบนข้อมูลเข้ารหัส), และมีคำสั่ง `pdpg` สำหรับใช้งานผ่าน command line
(`keygen` / `encrypt` / `inspect` / `decrypt`)

## ขอบเขตความสามารถ

| | |
|---|---|
| ใช้ได้ทันที | `+ - * /ค่าคงที่` · `@` · `dot` · `sum` · `mean` · `square` · `**n` · sigmoid โดยประมาณ · เลือกคอลัมน์ · บันทึก/อ่านไฟล์ · โมเดลเชิงเส้น sklearn |
| ต้องเขียนใหม่แบบไม่มีเงื่อนไข (branchless) | ตรรกะที่ขึ้นกับค่าของข้อมูล เช่น `if/else` เขียนเป็น `gate*b + (1-gate)*c` โดยใช้ sigmoid เป็นเกต — รันได้วันนี้ |
| รอการพัฒนาของเอนจิน | การเปรียบเทียบแบบแม่นยำ · `sort`/`max` · การหารด้วยข้อมูลเข้ารหัส · `exp`/`log`/`sqrt` · bootstrapping (ความลึกไม่จำกัด) — เมื่อเอนจินรองรับ โค้ดผู้ใช้ไม่ต้องแก้ |

สิ่งที่เป็นไปไม่ได้โดยการออกแบบ (ไม่ใช่ข้อจำกัดชั่วคราว): การที่ฝั่งผู้ประมวลผล
*อ่านค่า* ของข้อมูล ไม่ว่าทางใด — ทุกการพยายามจะได้รับข้อความอธิบายว่าทำไม
และควรทำอย่างไรแทน

## ข้อจำกัดที่ควรทราบ

- CKKS เป็นเลขคณิตแบบประมาณค่า (คลาดเคลื่อน ~1e-4 ในเวิร์กโหลดตัวอย่าง)
- ความลึกของการคูณจำกัดที่ 4 ชั้น เกินกว่านั้นจะมีข้อความแจ้งชัดเจน
- อาร์เรย์ 1–2 มิติ สูงสุด 8,192 แถวต่อชุด
- ไฟล์เข้ารหัสมีขนาดราว 1 MB ต่อคอลัมน์
- **ข้อมูลที่เข้ารหัสยังคงเป็นข้อมูลส่วนบุคคล** ตาม PDPA และ GDPR
  (เป็นการทำให้เป็นนามแฝง ไม่ใช่การทำให้เป็นนิรนาม) — pypdpg
  เป็นมาตรการเชิงเทคนิคเพื่อลดความเสี่ยง ไม่ได้ทำให้ข้อมูลพ้นจากกฎหมาย

รายละเอียดเชิงลึก (สถาปัตยกรรม การวัดผล ขอบเขตความปลอดภัย) อยู่ใน
[docs/design.md](docs/design.md) และ [docs/fine-print.md](docs/fine-print.md)
(ภาษาอังกฤษ)

## โน้ตบุ๊กตัวอย่าง

| โน้ตบุ๊ก | เนื้อหา |
|---|---|
| [getting_started](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/getting_started.ipynb) | เวิร์กโฟลว์สองฝ่ายครบวงจร: เข้ารหัส คำนวณแบบมองไม่เห็นข้อมูล ถอดรหัส |
| [01 · sklearn](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/cookbook/01_sklearn_models.ipynb) | โมเดลที่ฝึกแล้วคำนวณบนข้อมูลเข้ารหัส รวมถึงการแบ่งกลุ่มด้วย KMeans |
| [02 · dataframes](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/cookbook/02_encrypted_dataframes.ipynb) | คอลัมน์มีชื่อ สถิติรายคอลัมน์ โดยข้อมูลเข้ารหัสตลอด |
| [03 · branchless](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/cookbook/03_branchless_logic.ipynb) | สิ่งที่ระบบปฏิเสธ และวิธีเขียนใหม่ให้คำนวณได้โดยไม่เปิดเผยข้อมูล |
| [04 · CLI](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/cookbook/04_cli_workflow.ipynb) | ใช้งานผ่าน command line ทั้งวงจร รวมถึงมุมมองของผู้ดักไฟล์ |

## สัญญาอนุญาต

[PDPG Community License](LICENSE.md) (source-available) — ใช้ฟรีสำหรับบุคคลทั่วไป
สถานศึกษา งานวิจัย องค์กรไม่แสวงกำไร หน่วยงานภาครัฐ
และองค์กรที่มีรายได้ต่อปีไม่เกิน 50 ล้านบาท (นิติบุคคลไทย) หรือ 1 ล้านดอลลาร์สหรัฐ
(อื่น ๆ) — องค์กรที่ใหญ่กว่านั้นต้องมีสัญญาอนุญาตระดับองค์กร ติดต่อได้ที่
[pdpglab.xyz](https://pdpglab.xyz)

---

ดูแลโดย [PDPG-lab](https://pdpglab.xyz) · เอกสารฉบับเต็ม: [README.md](README.md) (ภาษาอังกฤษ)
