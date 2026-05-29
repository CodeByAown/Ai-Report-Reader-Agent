import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';
import OpenAI from 'openai';
import { extractFile } from '@kreuzberg/node';
import 'dotenv/config';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;

// Ensure uploads directory exists (it's gitignored so won't be in the repo)
await fs.mkdir(path.join(__dirname, 'uploads'), { recursive: true });

if (!process.env.OPENAI_API_KEY) {
  console.error('ERROR: OPENAI_API_KEY environment variable is not set.');
  console.error('  Local: add it to your .env file');
  console.error('  Railway/Render: add it in the Variables tab of your service');
  process.exit(1);
}

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const SUPPORTED_TYPES = [
  'application/pdf',
  'image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/bmp', 'image/webp', 'image/gif',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'application/msword',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
];

const IMAGE_TYPES = new Set([
  'image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/bmp', 'image/webp', 'image/gif',
]);

const upload = multer({
  dest: path.join(__dirname, 'uploads'),
  limits: { fileSize: 50 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (SUPPORTED_TYPES.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error(`File type "${file.mimetype}" is not supported. Please upload a PDF, image (JPG/PNG/TIFF/BMP), or Word document.`));
    }
  },
});

// ─── Refusal Detection ──────────────────────────────────────────────────────

const REFUSAL_PATTERNS = [
  "i'm sorry, i can't",
  "i'm sorry, but i can't",
  "sorry, i can't assist",
  "i cannot assist with that",
  "i'm not able to",
  "i can't help with",
  "i cannot help with",
  "i'm unable to process",
  "i apologize, but i can't",
  "i apologize, but i cannot",
  "i can't extract",
  "i cannot extract",
  "unable to assist",
  "i won't be able to",
  "that's not something i can",
];

function isRefusal(text) {
  const lower = (text || '').toLowerCase().trim();
  return REFUSAL_PATTERNS.some(p => lower.includes(p));
}

// ─── Vision OCR with Retry Logic ────────────────────────────────────────────

const OCR_ATTEMPTS = [
  {
    system: `You are MedOCR, a specialized optical character recognition engine integrated into Neural Hub, a hospital document management system. You are used exclusively by licensed medical professionals. Your only function is to transcribe text from medical documents with maximum accuracy. You do not provide medical advice or diagnoses — you only transcribe what is written.`,
    user: `Please transcribe all text visible in this medical document. This includes:
- All printed and typed text
- All handwritten notes, values, and annotations (do your best even if handwriting is unclear)
- Every number, measurement, lab value, date, and identifier
- Table contents, column headers, and row labels
- Any stamps, footers, or headers

If any handwriting is illegible, write [unclear] for that portion. Return only the transcribed text, maintaining the document's structure as much as possible.`,
  },
  {
    system: null,
    user: `I am a doctor reviewing this medical document in our clinical system. Please transcribe all the text you can see in this image — including any handwritten notes, printed text, numbers, and labels. This is for our electronic health record system.`,
  },
  {
    system: null,
    user: `List all text visible in this image. Include every word, number, date, and character you can read — both printed and handwritten content. Organize it by the sections you see.`,
  },
];

async function performVisionOCR(filePath, mimeType, attempt = 0) {
  if (attempt >= OCR_ATTEMPTS.length) {
    throw new Error(
      'The AI was unable to process this image after multiple attempts. ' +
      'Please ensure the image is clear and well-lit, then try again. ' +
      'For very dark or blurry images, try enhancing the contrast first.'
    );
  }

  const data = await fs.readFile(filePath);
  const base64 = data.toString('base64');
  const { system, user } = OCR_ATTEMPTS[attempt];

  const messages = system
    ? [
        { role: 'system', content: system },
        {
          role: 'user',
          content: [
            { type: 'text', text: user },
            { type: 'image_url', image_url: { url: `data:${mimeType};base64,${base64}`, detail: 'high' } },
          ],
        },
      ]
    : [
        {
          role: 'user',
          content: [
            { type: 'text', text: user },
            { type: 'image_url', image_url: { url: `data:${mimeType};base64,${base64}`, detail: 'high' } },
          ],
        },
      ];

  let content = '';
  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages,
      max_tokens: 4096,
      temperature: 0.1,
    });
    content = response.choices[0].message.content || '';
  } catch (err) {
    throw err;
  }

  if (isRefusal(content)) {
    console.warn(`  OCR attempt ${attempt + 1} refused — retrying with simpler prompt...`);
    return performVisionOCR(filePath, mimeType, attempt + 1);
  }

  return content;
}

// ─── Kreuzberg Text Extraction ───────────────────────────────────────────────

async function extractWithKreuzberg(filePath, mimeType) {
  try {
    const result = await extractFile(filePath, mimeType, { disableOcr: true });
    return result.content || '';
  } catch (err) {
    console.warn('  Kreuzberg extraction warning:', err.message);
    return '';
  }
}

// ─── Medical Summary Generation ──────────────────────────────────────────────

async function generateMedicalSummary(reportText) {
  const response = await openai.chat.completions.create({
    model: 'gpt-4o',
    messages: [
      {
        role: 'system',
        content: `You are an advanced clinical AI assistant embedded in Neural Hub, a medical intelligence platform used by licensed physicians. Your role is to analyze extracted medical document text and produce comprehensive, clinically accurate structured summaries with intelligent medical reasoning.

MANDATORY RULES — you must follow these without exception:

RULE 1 — NEVER leave "diagnosis", "medications", or "recommendations" empty or as "Not stated":
  • diagnosis: Extract from report if present. If absent, perform clinical reasoning based on the findings, lab abnormalities, and symptoms to provide a likely differential assessment. Prefix inferred text with "[AI Assessment]".
  • medications: Extract from report if present. If absent, based on the diagnosis/findings, suggest likely medication categories, drug classes, or treatment approaches clinically appropriate for the condition. Prefix each inferred item with "[AI Suggested]".
  • recommendations: Extract from report if present. If absent, generate 4–6 clinically appropriate recommendations: follow-up tests, specialist referrals, lifestyle modifications, dietary advice, monitoring instructions, and warning signs. Prefix each inferred item with "[AI Recommended]".

RULE 2 — Always populate "aiInferredFields" with the names of any fields where you inferred/generated content rather than extracting it directly from the report. Possible values: "diagnosis", "medications", "recommendations".

RULE 3 — Write a rich, detailed "summary" of 4–6 sentences:
  • Sentence 1: Identify the report type, patient context, and overall clinical picture.
  • Sentences 2–3: Explicitly state what findings are NORMAL and what findings are ABNORMAL or elevated/decreased, with specific values where available.
  • Sentence 4: State the clinical significance, urgency level, and potential conditions suggested by the abnormalities.
  • Sentences 5–6: Describe next steps, follow-up actions, and any critical concerns that require immediate attention.

RULE 4 — "keyFindings" must contain at least 3 items. For each abnormal value, include the value and its clinical significance. For normal panels, confirm specific tests were within range.

RULE 5 — "criticalValues" should include any value that is significantly outside normal range, not just values the report explicitly flags — use clinical knowledge to identify them.

Respond with a valid JSON object using EXACTLY these fields:
{
  "patientInfo": {
    "name": "full name or 'Not found'",
    "ageOrDob": "age or date of birth or 'Not found'",
    "id": "patient ID / MRN / record number or 'Not found'",
    "gender": "gender or 'Not found'",
    "reportDate": "date of report/test or 'Not found'"
  },
  "reportType": "specific report type (e.g., 'Complete Blood Count', 'Chest X-Ray', 'MRI Brain', 'Prescription', 'Discharge Summary', 'Pathology Report', 'Handwritten Clinical Note')",
  "keyFindings": ["at least 3 findings as complete clinical sentences, each describing the finding AND its significance"],
  "diagnosis": "extracted diagnosis from report, OR [AI Assessment]: inferred differential diagnosis based on findings",
  "criticalValues": ["any clinically significant abnormal values — include the value, unit, and normal reference range if known"],
  "labValues": [
    {"test": "test name", "value": "result", "unit": "unit or ''", "status": "normal | high | low | critical | unknown"}
  ],
  "medications": ["medications from report, OR [AI Suggested] drug class / treatment approach for the condition"],
  "recommendations": ["recommendations from report, OR [AI Recommended] clinical follow-up, lifestyle, dietary, and monitoring recommendations"],
  "summary": "detailed 4–6 sentence clinical narrative: report overview → what is normal → what is abnormal with values → clinical significance → next steps and critical actions",
  "urgencyLevel": "NORMAL | ATTENTION | URGENT",
  "dataQuality": "COMPLETE | PARTIAL | LIMITED",
  "aiInferredFields": ["array of field names where content was AI-inferred, not extracted — e.g., ['diagnosis','medications','recommendations']"]
}

urgencyLevel rules:
- URGENT: life-threatening values, critical results, phrases like 'immediate', 'emergency', 'stat', or any value at severe risk thresholds
- ATTENTION: abnormal values, H/L flags, phrases like 'follow up', 'refer to', abnormal imaging, moderately elevated markers
- NORMAL: routine results within normal range, normal imaging, prescription refills

dataQuality rules:
- COMPLETE: report text is clear and all major sections readable
- PARTIAL: some sections unclear or handwriting partially illegible
- LIMITED: very short text, mostly illegible, or incomplete document

Always populate labValues from any numbers/values you can identify. Use clinical knowledge to assign status (high/low/critical/normal) even if the report does not explicitly flag them.`,
      },
      {
        role: 'user',
        content: `Analyze this medical document text and return the structured JSON summary with full clinical reasoning:\n\n${reportText.substring(0, 15000)}`,
      },
    ],
    response_format: { type: 'json_object' },
    max_tokens: 3500,
    temperature: 0.15,
  });

  return JSON.parse(response.choices[0].message.content);
}

// ─── Routes ───────────────────────────────────────────────────────────────────

app.use(express.static(path.join(__dirname, 'public')));

app.get('/health', (_req, res) => res.json({ status: 'ok', service: 'Neural Hub Medical Agent', version: '3.0.0' }));

app.post('/analyze', upload.single('report'), async (req, res) => {
  const file = req.file;
  if (!file) return res.status(400).json({ error: 'No file uploaded.' });

  const tempPath = file.path;

  try {
    let extractedText = '';
    let extractionMethod = 'digital';
    const isImage = IMAGE_TYPES.has(file.mimetype);

    console.log(`\n  Analyzing: ${file.originalname} (${file.mimetype})`);

    if (isImage) {
      console.log('  Step 1: Vision OCR...');
      extractedText = await performVisionOCR(tempPath, file.mimetype);
      extractionMethod = 'vision-ocr';
    } else {
      console.log('  Step 1: Digital text extraction...');
      extractedText = await extractWithKreuzberg(tempPath, file.mimetype);

      if (extractedText.trim().length < 120) {
        console.log('  Step 1b: Insufficient text — attempting vision OCR for scanned document...');
        try {
          extractedText = await performVisionOCR(tempPath, 'image/jpeg');
          extractionMethod = 'vision-ocr';
        } catch (visionErr) {
          if (extractedText.trim().length > 10) {
            extractionMethod = 'partial-digital';
          } else {
            return res.status(422).json({
              error: 'This appears to be a scanned PDF with no selectable text. For best results, please export each page as a JPG or PNG image and upload that instead.',
            });
          }
        }
      }
    }

    if (!extractedText.trim()) {
      return res.status(422).json({
        error: 'Could not extract readable text from this document. Please check the file quality and try again.',
      });
    }

    console.log(`  Step 2: Extracted ${extractedText.length} characters (${extractionMethod})`);
    console.log('  Step 3: Generating medical summary with clinical reasoning...');

    const summary = await generateMedicalSummary(extractedText);
    console.log(`  Done. Urgency: ${summary.urgencyLevel}, Quality: ${summary.dataQuality}, Inferred: ${(summary.aiInferredFields || []).join(', ') || 'none'}`);

    res.json({
      success: true,
      fileName: file.originalname,
      fileSize: file.size,
      mimeType: file.mimetype,
      extractionMethod,
      extractedText: extractedText.substring(0, 10000),
      summary,
      processedAt: new Date().toISOString(),
    });
  } catch (err) {
    console.error('  Error:', err.message);
    res.status(500).json({ error: err.message || 'Analysis failed. Please try again.' });
  } finally {
    await fs.unlink(tempPath).catch(() => {});
  }
});

// Multer error handler
app.use((err, _req, res, _next) => {
  console.error('Request error:', err.message);
  res.status(400).json({ error: err.message || 'Request failed.' });
});

app.listen(PORT, () => {
  console.log('\n  ┌─────────────────────────────────────────┐');
  console.log('  │                                         │');
  console.log('  │   🧠  Neural Hub — Medical Agent v3.0   │');
  console.log(`  │   Running at: http://localhost:${PORT}      │`);
  console.log('  │                                         │');
  console.log('  └─────────────────────────────────────────┘\n');
});
