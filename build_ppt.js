const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const ROOT = "C:\\Users\\Manas Kushwaha\\Desktop\\Cyberbullying Detector plus Safe Reply Generator";

function img(rel) {
  return path.join(ROOT, "figures", rel);
}

let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "FYP Team";
pres.title = "Cyberbullying Detection and Safe Reply Generation";
pres.subject = "Final Year Research Project - End-term Presentation";

// Color palette based on template
const C = {
  brown: "572314",
  blue: "0070C0",
  teal: "3891A7",
  beige: "B5A989",
  black: "000000",
  darkGray: "333333",
  white: "FFFFFF",
  lightGray: "F2F2F2",
  tableHeader: "E7E6E6",
  tableBorder: "BFBFBF",
};

const F = {
  title: "Times New Roman",
  body: "Gill Sans MT",
  alt: "Arial",
  boldTitle: "Bahnschrift SemiBold",
};

function addPageNumber(slide, num) {
  slide.addText(String(num), {
    x: 9.0, y: 5.15, w: 0.8, h: 0.3,
    fontSize: 10, fontFace: F.body, color: C.beige,
    align: "center", valign: "middle"
  });
}

function titleShape(slide, text, y = 0.15) {
  slide.addText(text, {
    x: 0.6, y, w: 9, h: 0.6,
    fontSize: 32, fontFace: F.title, color: C.brown, bold: true,
    align: "left", valign: "middle"
  });
}

function subtitleShape(slide, text, y = 0.75) {
  slide.addText(text, {
    x: 0.6, y, w: 9, h: 0.35,
    fontSize: 14, fontFace: F.body, color: C.teal,
    align: "left", valign: "middle"
  });
}

// ==========================
// Slide 1: Title Slide
// ==========================
let s1 = pres.addSlide();
s1.background = { color: C.white };
s1.addText("FINAL YEAR RESEARCH PROJECT", {
  x: 0.6, y: 0.8, w: 8.5, h: 0.5,
  fontSize: 20, fontFace: F.boldTitle, color: C.darkGray,
  align: "left", valign: "middle"
});
s1.addText("End-term Presentation", {
  x: 0.6, y: 1.25, w: 8.5, h: 0.5,
  fontSize: 28, fontFace: F.boldTitle, color: C.darkGray,
  align: "left", valign: "middle"
});
s1.addText("Cyberbullying Detection and Safe Reply Generation", {
  x: 0.6, y: 2.0, w: 8.5, h: 0.6,
  fontSize: 36, fontFace: "Bebas Neue", color: C.brown,
  align: "left", valign: "middle"
});
s1.addText("Supervised By: Dr. Shrabanee Swagatika", {
  x: 0.6, y: 3.3, w: 6, h: 0.3,
  fontSize: 14, fontFace: F.body, color: C.blue, bold: true,
  align: "left", valign: "middle"
});
s1.addText("Group No.: 1\nName of the Student(s) with Regd. No.:\nSunny Nonia - 2241014158\nReshav Kumar Choudhary - 2241003036\nManas Kushwaha - 2241011240\nAbhishek Pati - 2241016236", {
  x: 0.6, y: 3.7, w: 6, h: 1.2,
  fontSize: 12, fontFace: F.body, color: C.black, bold: true,
  align: "left", valign: "top"
});
s1.addText("Department of Computer Sc. and Engineering\nFaculty of Engineering & Technology (ITER)\nSiksha 'O' Anusandhan (Deemed to be) University\nBhubaneswar, Odisha", {
  x: 5.5, y: 4.2, w: 4.2, h: 1.0,
  fontSize: 11, fontFace: F.body, color: C.black, bold: true,
  align: "right", valign: "bottom"
});
addPageNumber(s1, 1);

// ==========================
// Slide 2: Presentation Outline
// ==========================
let s2 = pres.addSlide();
s2.background = { color: C.white };
titleShape(s2, "Presentation Outline");
let outlineItems = [
  { text: "Problem Statement & Motivations", size: 16, indent: 0 },
  { text: "Problem Statement", size: 14, indent: 1 },
  { text: "Motivations", size: 14, indent: 1 },
  { text: "Literature Review", size: 16, indent: 0 },
  { text: "Existing Solutions & Limitations", size: 14, indent: 1 },
  { text: "Proposed Methodology & Approach", size: 16, indent: 0 },
  { text: "Key Components/Modules & Algorithms Used", size: 16, indent: 0 },
  { text: "Dataset(s) & Tools/Technologies", size: 16, indent: 0 },
  { text: "Experimentation", size: 16, indent: 0 },
  { text: "System Specification", size: 14, indent: 1 },
  { text: "Experimental Results", size: 14, indent: 1 },
  { text: "Performance Analysis", size: 14, indent: 1 },
  { text: "Conclusion and Future Scope", size: 16, indent: 0 },
  { text: "References", size: 16, indent: 0 },
];
let outlineText = outlineItems.map(it => ({
  text: it.text,
  options: { bullet: true, breakLine: true, fontSize: it.size, indentLevel: it.indent }
}));
s2.addText(outlineText, {
  x: 0.8, y: 1.0, w: 8.5, h: 4.5,
  fontFace: F.body, color: C.black,
  paraSpaceAfter: 4,
  bullet: { color: C.teal }
});
addPageNumber(s2, 2);

// ==========================
// Slide 3: Problem Statement
// ==========================
let s3 = pres.addSlide();
s3.background = { color: C.white };
titleShape(s3, "Problem Statement");
s3.addText([
  { text: "Cyberbullying is a growing concern on social media platforms, causing severe psychological harm to victims.", options: { breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "Key Challenges:", options: { breakLine: true, bold: true, fontSize: 18 } },
  { text: "• Lack of automated, real-time detection systems for multi-class cyberbullying categories.", options: { breakLine: true, bullet: true } },
  { text: "• Existing models struggle with contextual understanding and semantic nuances in textual data.", options: { breakLine: true, bullet: true } },
  { text: "• High computational cost of deep learning models makes deployment difficult.", options: { breakLine: true, bullet: true } },
  { text: "• No integrated safe-reply mechanism to de-escalate toxic conversations in real time.", options: { breakLine: true, bullet: true } },
  { text: "", options: { breakLine: true } },
  { text: "Objective: Build an end-to-end pipeline that detects cyberbullying across ethnicity/race, gender/sexual, and religion categories, and generates safe, empathetic replies to counter toxicity.", options: { breakLine: true, fontSize: 15, italic: true } }
], {
  x: 0.7, y: 1.0, w: 8.8, h: 4.2,
  fontFace: F.body, color: C.black, fontSize: 16,
  paraSpaceAfter: 6
});
addPageNumber(s3, 3);

// ==========================
// Slide 4: Motivations
// ==========================
let s4 = pres.addSlide();
s4.background = { color: C.white };
titleShape(s4, "Motivations");
s4.addText([
  { text: "Rising Cyberbullying Incidents:", options: { breakLine: true, bold: true } },
  { text: "Over 40% of internet users report experiencing some form of online harassment.", options: { breakLine: true, bullet: true } },
  { text: "", options: { breakLine: true } },
  { text: "Mental Health Impact:", options: { breakLine: true, bold: true } },
  { text: "Victims suffer from anxiety, depression, and in severe cases, self-harm or suicidal ideation.", options: { breakLine: true, bullet: true } },
  { text: "", options: { breakLine: true } },
  { text: "Platform Responsibility:", options: { breakLine: true, bold: true } },
  { text: "Social media platforms need scalable AI solutions to moderate content and protect users.", options: { breakLine: true, bullet: true } },
  { text: "", options: { breakLine: true } },
  { text: "Research Gap:", options: { breakLine: true, bold: true } },
  { text: "Most existing work focuses only on binary detection; multi-class contextual detection with safe reply generation is underexplored.", options: { breakLine: true, bullet: true } },
  { text: "", options: { breakLine: true } },
  { text: "Technical Motivation:", options: { breakLine: true, bold: true } },
  { text: "Leverage transformer-based models (DistilBERT) and parameter-efficient fine-tuning (LoRA) for high accuracy with reduced compute.", options: { breakLine: true, bullet: true } }
], {
  x: 0.7, y: 1.0, w: 8.8, h: 4.2,
  fontFace: F.body, color: C.black, fontSize: 16,
  paraSpaceAfter: 4
});
addPageNumber(s4, 4);

// ==========================
// Slide 5: Literature Review
// ==========================
let s5 = pres.addSlide();
s5.background = { color: C.white };
titleShape(s5, "Literature Review");
subtitleShape(s5, "Summary of Existing Approaches and Research Limitations");

let litRows = [
  [
    { text: "S. No.", options: { bold: true, fill: { color: C.tableHeader }, color: C.black } },
    { text: "Author & Year", options: { bold: true, fill: { color: C.tableHeader }, color: C.black } },
    { text: "Methods/Techniques", options: { bold: true, fill: { color: C.tableHeader }, color: C.black } },
    { text: "Dataset Used", options: { bold: true, fill: { color: C.tableHeader }, color: C.black } },
    { text: "Research Outcomes", options: { bold: true, fill: { color: C.tableHeader }, color: C.black } },
    { text: "Limitations", options: { bold: true, fill: { color: C.tableHeader }, color: C.black } }
  ],
  ["1", "Gutierrez-Batista et al. (2024)", "Sentence-BERT fine-tuned + SVM/LightGBM", "BullyingV3.0, MySpace, Hate-speech", "Acc: 83-97%", "No multimodal support; no intervention/response generation"],
  ["2", "Muneer et al. (2023)", "Stacked ensemble: Conv1D-LSTM, BiLSTM, CNN + Tuned BERT", "Twitter (37K tweets)", "Acc: 97.4%, F1: 0.964", "Poor cross-platform generalization; no intervention"],
  ["3", "Yi & Zubiaga (2022)", "Transformer + adversarial domain adaptation", "Formspring, Twitter, Instagram", "Significant improvement over naive transfer", "Binary only; requires unlabeled target data; no intervention"],
  ["4", "Tabassum & Nunavath (2024)", "RoBERTa + ViT multi-modal late fusion", "Public meme datasets", "Acc: 99.2%, F1: 0.992", "Small private dataset; English-only; no response generation"],
  ["5", "Garcia-Mendez & De Arriba-Perez (2024)", "Adaptive Random Forest + GPT-4o-mini reasoning", "Kaggle Twitter (15,890 posts)", "Acc: 90.55%, F1: 90.06%", "Binary only; expensive LLM reasoning; no automated intervention"],
  ["6", "Sanh et al. (2019)", "Knowledge distillation into DistilBERT", "GLUE benchmark", "Retains 97% of BERT performance", "Not evaluated specifically for cyberbullying tasks"],
  ["7", "Hu et al. (2022)", "Low-Rank Adaptation (LoRA) of transformers", "GPT-3, RoBERTa, BERT benchmarks", "Comparable to full fine-tuning with fewer trainable params", "Limited evaluation on cyberbullying tasks"],
  ["8", "Proposed Work (2026)", "DistilBERT + LoRA + DistilGPT-2", "Cyberbullying Multi-label (99,990)", "Acc: 99.74%, F1: 0.997", "High accuracy + parameter efficiency + safe reply generation"]
];

s5.addTable(litRows, {
  x: 0.3, y: 1.25, w: 9.5, h: 4.0,
  fontFace: F.alt, fontSize: 9,
  border: { pt: 0.5, color: C.tableBorder },
  colW: [0.6, 1.5, 2.2, 1.6, 1.5, 1.9],
  valign: "middle", align: "center",
  autoPage: false
});
addPageNumber(s5, 5);

// ==========================
// Slide 6: Research Gap & Improvements
// ==========================
let s6 = pres.addSlide();
s6.background = { color: C.white };
titleShape(s6, "Research Gap & Improvements");
s6.addText([
  { text: "Identified Gaps in Existing Work:", options: { breakLine: true, bold: true, fontSize: 18, color: C.brown } },
  { text: "", options: { breakLine: true } },
  { text: "1. Limited Integrated Detection and Intervention Pipelines: Most works focus only on detection; few combine detection with automated safe-response generation.", options: { breakLine: true, bullet: true } },
  { text: "", options: { breakLine: true } },
  { text: "2. Lack of Safety-Qualified Automated Response Systems: Few systems evaluate generated interventions using measurable safety metrics (toxicity, empathy, relevance).", options: { breakLine: true, bullet: true } },
  { text: "", options: { breakLine: true } },
  { text: "3. Limited Interpretable Evaluation: Existing works primarily evaluate classification accuracy; less focus on interpretable frameworks for generated moderation responses.", options: { breakLine: true, bullet: true } },
  { text: "", options: { breakLine: true } },
  { text: "4. Resource-Efficient Detection is Underexplored: Limited work evaluates lightweight transformers (DistilBERT) and parameter-efficient methods (LoRA) for cyberbullying.", options: { breakLine: true, bullet: true } },
  { text: "", options: { breakLine: true } },
  { text: "Our Improvements:", options: { breakLine: true, bold: true, fontSize: 18, color: C.brown } },
  { text: "", options: { breakLine: true } },
  { text: "• Multi-class classification across 4 categories with >99.7% accuracy.", options: { breakLine: true, bullet: true } },
  { text: "• Parameter-efficient fine-tuning using LoRA (0.59M trainable params vs 66.96M).", options: { breakLine: true, bullet: true } },
  { text: "• Integrated DistilGPT-2 based safe reply generator with toxicity-aware decoding.", options: { breakLine: true, bullet: true } },
  { text: "• Comprehensive evaluation via confusion matrices, McNemar test, and efficiency benchmarking.", options: { breakLine: true, bullet: true } }
], {
  x: 0.6, y: 1.0, w: 8.8, h: 4.3,
  fontFace: F.body, color: C.black, fontSize: 14,
  paraSpaceAfter: 1
});
addPageNumber(s6, 6);

// ==========================
// Slide 7: Proposed Solution & Architecture
// ==========================
let s7 = pres.addSlide();
s7.background = { color: C.white };
titleShape(s7, "Proposed Solution & Architecture");
s7.addImage({ path: img("01_system_architecture.png"), x: 1.5, y: 1.0, w: 7.2, h: 3.9, sizing: { type: "contain" } });
s7.addText("Fig. 1: System Architecture of the Proposed Cyberbullying Detection and Safe Reply Generation Pipeline", {
  x: 0.5, y: 5.0, w: 9, h: 0.3,
  fontSize: 11, fontFace: F.body, color: C.darkGray, italic: true,
  align: "center", valign: "middle"
});
addPageNumber(s7, 7);

// ==========================
// Slide 8: Dataset & Tools/Technologies
// ==========================
let s8 = pres.addSlide();
s8.background = { color: C.white };
titleShape(s8, "Dataset & Tools/Technologies");

// Left: Dataset stats
s8.addText("Dataset Statistics", {
  x: 0.6, y: 1.0, w: 4.2, h: 0.35,
  fontSize: 18, fontFace: F.body, color: C.brown, bold: true,
  align: "left", valign: "middle"
});

let dsRows = [
  [{ text: "Statistic", options: { bold: true, fill: { color: C.tableHeader } } }, { text: "Value", options: { bold: true, fill: { color: C.tableHeader } } }],
  ["Total Samples", "99,990"],
  ["Train Samples (80%)", "~79,992"],
  ["Validation Samples (10%)", "~9,999"],
  ["Test Samples (10%)", "~9,999"],
  ["Number of Classes", "4"],
  ["Classes", "not_cyberbullying, ethnicity/race, gender/sexual, religion"]
];
s8.addTable(dsRows, {
  x: 0.6, y: 1.4, w: 4.2, h: 2.2,
  fontFace: F.alt, fontSize: 11,
  border: { pt: 0.5, color: C.tableBorder },
  colW: [2.0, 2.2],
  valign: "middle"
});

s8.addImage({ path: img("02_class_distribution.png"), x: 5.2, y: 1.0, w: 4.2, h: 2.6, sizing: { type: "contain" } });

// Bottom: Tools
s8.addText("Tools & Technologies", {
  x: 0.6, y: 3.7, w: 4.2, h: 0.3,
  fontSize: 18, fontFace: F.body, color: C.brown, bold: true,
  align: "left", valign: "middle"
});
s8.addText([
  { text: "Languages & Libraries: Python, PyTorch, Transformers (Hugging Face), PEFT, scikit-learn, pandas, numpy", options: { breakLine: true, bullet: true } },
  { text: "Models: DistilBERT (detection), DistilGPT-2 (reply generation)", options: { breakLine: true, bullet: true } },
  { text: "Hardware: NVIDIA GeForce RTX 3060 Ti (8GB VRAM), CUDA 13.2", options: { breakLine: true, bullet: true } },
  { text: "Environment: Windows, Python 3.12", options: { breakLine: true, bullet: true } }
], {
  x: 0.6, y: 4.05, w: 8.8, h: 1.1,
  fontFace: F.body, color: C.black, fontSize: 13,
  paraSpaceAfter: 2,
  bullet: { color: C.teal }
});
addPageNumber(s8, 8);

// ==========================
// Slide 9: Experimentation - System Specification
// ==========================
let s9 = pres.addSlide();
s9.background = { color: C.white };
titleShape(s9, "Experimentation");
subtitleShape(s9, "System Specification");

let sysRows = [
  [{ text: "Component", options: { bold: true, fill: { color: C.tableHeader } } }, { text: "Specification", options: { bold: true, fill: { color: C.tableHeader } } }],
  ["Operating System", "Windows 11"],
  ["Processor", "AMD / Intel (x64)"],
  ["GPU", "NVIDIA GeForce RTX 3060 Ti"],
  ["GPU Memory", "8 GB GDDR6"],
  ["CUDA Version", "13.2"],
  ["Driver Version", "596.21"],
  ["RAM", "16+ GB"],
  ["Python Version", "3.12"],
  ["Deep Learning Framework", "PyTorch 2.x + Transformers 5.8.1"]
];
s9.addTable(sysRows, {
  x: 2.0, y: 1.4, w: 6.0, h: 3.2,
  fontFace: F.alt, fontSize: 13,
  border: { pt: 0.5, color: C.tableBorder },
  colW: [2.8, 3.2],
  valign: "middle", align: "left"
});
addPageNumber(s9, 9);

// ==========================
// Slide 10: Experimental Results - Performance Metrics
// ==========================
let s10 = pres.addSlide();
s10.background = { color: C.white };
titleShape(s10, "Experimental Results");
subtitleShape(s10, "Performance Metrics Comparison");

let perfRows = [
  [
    { text: "Model", options: { bold: true, fill: { color: C.tableHeader } } },
    { text: "Accuracy", options: { bold: true, fill: { color: C.tableHeader } } },
    { text: "Precision (Macro)", options: { bold: true, fill: { color: C.tableHeader } } },
    { text: "Recall (Macro)", options: { bold: true, fill: { color: C.tableHeader } } },
    { text: "F1-Score (Macro)", options: { bold: true, fill: { color: C.tableHeader } } },
    { text: "F1-Score (Weighted)", options: { bold: true, fill: { color: C.tableHeader } } }
  ],
  ["TF-IDF+SVM", "0.9940", "0.9942", "0.9920", "0.9931", "0.9940"],
  ["LSTM", "0.9924", "0.9932", "0.9905", "0.9918", "0.9924"],
  ["DistilBERT", "0.9974", "0.9976", "0.9962", "0.9969", "0.9974"],
  ["DistilBERT+LoRA", "0.9973", "0.9973", "0.9964", "0.9969", "0.9973"]
];
s10.addTable(perfRows, {
  x: 0.4, y: 1.3, w: 9.3, h: 1.8,
  fontFace: F.alt, fontSize: 12,
  border: { pt: 0.5, color: C.tableBorder },
  colW: [1.8, 1.3, 1.8, 1.5, 1.6, 1.7],
  valign: "middle", align: "center"
});

s10.addText("Key Observations:", {
  x: 0.6, y: 3.3, w: 8.8, h: 0.3,
  fontSize: 15, fontFace: F.body, color: C.brown, bold: true,
  align: "left", valign: "middle"
});
s10.addText([
  { text: "DistilBERT achieves the highest accuracy (99.74%) and F1-score among all models.", options: { breakLine: true, bullet: true } },
  { text: "DistilBERT+LoRA matches DistilBERT performance with only 0.88% trainable parameters.", options: { breakLine: true, bullet: true } },
  { text: "TF-IDF+SVM provides strong baseline performance with minimal training time (3.12s).", options: { breakLine: true, bullet: true } }
], {
  x: 0.6, y: 3.65, w: 8.8, h: 1.2,
  fontFace: F.body, color: C.black, fontSize: 13,
  paraSpaceAfter: 3,
  bullet: { color: C.teal }
});
addPageNumber(s10, 10);

// ==========================
// Slide 11: Confusion Matrices (Validation - Part 1)
// ==========================
let s11 = pres.addSlide();
s11.background = { color: C.white };
titleShape(s11, "Experimental Results");
subtitleShape(s11, "Confusion Matrices - Validation Set (Part 1)");

s11.addImage({ path: img("05_confusion_matrix_tfidf_svm.png"), x: 0.5, y: 1.2, w: 4.2, h: 3.2, sizing: { type: "contain" } });
s11.addText("TF-IDF + SVM", { x: 0.5, y: 4.45, w: 4.2, h: 0.2, fontSize: 12, fontFace: F.body, align: "center", color: C.darkGray, bold: true });

s11.addImage({ path: img("06_confusion_matrix_lstm.png"), x: 5.3, y: 1.2, w: 4.2, h: 3.2, sizing: { type: "contain" } });
s11.addText("LSTM", { x: 5.3, y: 4.45, w: 4.2, h: 0.2, fontSize: 12, fontFace: F.body, align: "center", color: C.darkGray, bold: true });

s11.addText("Fig. 2(a): Validation Confusion Matrices for TF-IDF+SVM and LSTM", {
  x: 0.5, y: 4.8, w: 9, h: 0.2,
  fontSize: 11, fontFace: F.body, color: C.darkGray, italic: true,
  align: "center", valign: "middle"
});
addPageNumber(s11, 11);

// ==========================
// Slide 12: Confusion Matrices (Validation - Part 2)
// ==========================
let s12 = pres.addSlide();
s12.background = { color: C.white };
titleShape(s12, "Experimental Results");
subtitleShape(s12, "Confusion Matrices - Validation Set (Part 2)");

s12.addImage({ path: img("07_confusion_matrix_distilbert.png"), x: 0.5, y: 1.2, w: 4.2, h: 3.2, sizing: { type: "contain" } });
s12.addText("DistilBERT", { x: 0.5, y: 4.45, w: 4.2, h: 0.2, fontSize: 12, fontFace: F.body, align: "center", color: C.darkGray, bold: true });

s12.addImage({ path: img("08_confusion_matrix_distilbert_lora.png"), x: 5.3, y: 1.2, w: 4.2, h: 3.2, sizing: { type: "contain" } });
s12.addText("DistilBERT + LoRA", { x: 5.3, y: 4.45, w: 4.2, h: 0.2, fontSize: 12, fontFace: F.body, align: "center", color: C.darkGray, bold: true });

s12.addText("Fig. 2(b): Validation Confusion Matrices for DistilBERT and DistilBERT+LoRA", {
  x: 0.5, y: 4.8, w: 9, h: 0.2,
  fontSize: 11, fontFace: F.body, color: C.darkGray, italic: true,
  align: "center", valign: "middle"
});
addPageNumber(s12, 12);

// ==========================
// Slide 13: Confusion Matrices (Test - Part 1)
// ==========================
let s13 = pres.addSlide();
s13.background = { color: C.white };
titleShape(s13, "Experimental Results");
subtitleShape(s13, "Confusion Matrices - Test Set (Part 1)");

s13.addImage({ path: img("confusion_matrix_tf_idf_svm_test.png"), x: 0.5, y: 1.2, w: 4.2, h: 3.2, sizing: { type: "contain" } });
s13.addText("TF-IDF + SVM", { x: 0.5, y: 4.45, w: 4.2, h: 0.2, fontSize: 12, fontFace: F.body, align: "center", color: C.darkGray, bold: true });

s13.addImage({ path: img("confusion_matrix_lstm_test.png"), x: 5.3, y: 1.2, w: 4.2, h: 3.2, sizing: { type: "contain" } });
s13.addText("LSTM", { x: 5.3, y: 4.45, w: 4.2, h: 0.2, fontSize: 12, fontFace: F.body, align: "center", color: C.darkGray, bold: true });

s13.addText("Fig. 3(a): Test Confusion Matrices for TF-IDF+SVM and LSTM", {
  x: 0.5, y: 4.8, w: 9, h: 0.2,
  fontSize: 11, fontFace: F.body, color: C.darkGray, italic: true,
  align: "center", valign: "middle"
});
addPageNumber(s13, 13);

// ==========================
// Slide 14: Confusion Matrices (Test - Part 2)
// ==========================
let s14 = pres.addSlide();
s14.background = { color: C.white };
titleShape(s14, "Experimental Results");
subtitleShape(s14, "Confusion Matrices - Test Set (Part 2)");

s14.addImage({ path: img("confusion_matrix_distilbert_test.png"), x: 0.5, y: 1.2, w: 4.2, h: 3.2, sizing: { type: "contain" } });
s14.addText("DistilBERT", { x: 0.5, y: 4.45, w: 4.2, h: 0.2, fontSize: 12, fontFace: F.body, align: "center", color: C.darkGray, bold: true });

s14.addImage({ path: img("confusion_matrix_distilbert_lora_test.png"), x: 5.3, y: 1.2, w: 4.2, h: 3.2, sizing: { type: "contain" } });
s14.addText("DistilBERT + LoRA", { x: 5.3, y: 4.45, w: 4.2, h: 0.2, fontSize: 12, fontFace: F.body, align: "center", color: C.darkGray, bold: true });

s14.addText("Fig. 3(b): Test Confusion Matrices for DistilBERT and DistilBERT+LoRA", {
  x: 0.5, y: 4.8, w: 9, h: 0.2,
  fontSize: 11, fontFace: F.body, color: C.darkGray, italic: true,
  align: "center", valign: "middle"
});
addPageNumber(s14, 14);

// ==========================
// Slide 15: Accuracy Comparison & Curves
// ==========================
let s15 = pres.addSlide();
s15.background = { color: C.white };
titleShape(s15, "Experimental Results");
subtitleShape(s15, "Accuracy Comparison & Validation Curves");

s15.addImage({ path: img("03_accuracy_comparison.png"), x: 0.4, y: 1.2, w: 4.5, h: 3.2, sizing: { type: "contain" } });
s15.addText("Model Accuracy Comparison", { x: 0.4, y: 4.45, w: 4.5, h: 0.2, fontSize: 11, fontFace: F.body, align: "center", color: C.darkGray, bold: true });

s15.addImage({ path: img("12_accuracy_curves.png"), x: 5.1, y: 1.2, w: 4.5, h: 3.2, sizing: { type: "contain" } });
s15.addText("Validation Accuracy Curves", { x: 5.1, y: 4.45, w: 4.5, h: 0.2, fontSize: 11, fontFace: F.body, align: "center", color: C.darkGray, bold: true });

s15.addText("Fig. 4: Accuracy comparison across models (left) and validation accuracy curves over epochs (right)", {
  x: 0.5, y: 4.8, w: 9, h: 0.2,
  fontSize: 11, fontFace: F.body, color: C.darkGray, italic: true,
  align: "center", valign: "middle"
});
addPageNumber(s15, 15);

// ==========================
// Slide 16: Loss Curves
// ==========================
let s16 = pres.addSlide();
s16.background = { color: C.white };
titleShape(s16, "Experimental Results");
subtitleShape(s16, "Training & Validation Loss Curves");

s16.addImage({ path: img("10_loss_curve_distilbert.png"), x: 0.3, y: 1.1, w: 3.0, h: 2.6, sizing: { type: "contain" } });
s16.addText("DistilBERT", { x: 0.3, y: 3.75, w: 3.0, h: 0.2, fontSize: 11, fontFace: F.body, align: "center", color: C.darkGray, bold: true });

s16.addImage({ path: img("11_loss_curve_distilbert_lora.png"), x: 3.5, y: 1.1, w: 3.0, h: 2.6, sizing: { type: "contain" } });
s16.addText("DistilBERT + LoRA", { x: 3.5, y: 3.75, w: 3.0, h: 0.2, fontSize: 11, fontFace: F.body, align: "center", color: C.darkGray, bold: true });

s16.addImage({ path: img("09_loss_curve_lstm.png"), x: 6.7, y: 1.1, w: 3.0, h: 2.6, sizing: { type: "contain" } });
s16.addText("LSTM", { x: 6.7, y: 3.75, w: 3.0, h: 0.2, fontSize: 11, fontFace: F.body, align: "center", color: C.darkGray, bold: true });

s16.addText("Fig. 5: Training and validation loss curves across epochs for DistilBERT, DistilBERT+LoRA, and LSTM", {
  x: 0.5, y: 4.1, w: 9, h: 0.2,
  fontSize: 11, fontFace: F.body, color: C.darkGray, italic: true,
  align: "center", valign: "middle"
});
addPageNumber(s16, 16);

// ==========================
// Slide 17: Efficiency Analysis
// ==========================
let s17 = pres.addSlide();
s17.background = { color: C.white };
titleShape(s17, "Experimental Results");
subtitleShape(s17, "Efficiency Analysis");

let effRows = [
  [
    { text: "Model", options: { bold: true, fill: { color: C.tableHeader } } },
    { text: "Training Time (s)", options: { bold: true, fill: { color: C.tableHeader } } },
    { text: "Inference Time (s)", options: { bold: true, fill: { color: C.tableHeader } } },
    { text: "Total Parameters", options: { bold: true, fill: { color: C.tableHeader } } },
    { text: "Trainable Parameters", options: { bold: true, fill: { color: C.tableHeader } } }
  ],
  ["TF-IDF+SVM", "3.12", "0.17", "N/A", "N/A"],
  ["LSTM", "147.82", "0.63", "N/A", "N/A"],
  ["DistilBERT", "3600.00", "14.56", "66.96M", "66.96M"],
  ["DistilBERT+LoRA", "2562.19", "15.46", "67.85M", "0.59M"]
];
s17.addTable(effRows, {
  x: 0.5, y: 1.2, w: 9.0, h: 1.6,
  fontFace: F.alt, fontSize: 12,
  border: { pt: 0.5, color: C.tableBorder },
  colW: [2.0, 1.8, 1.8, 1.8, 1.8],
  valign: "middle", align: "center"
});

s17.addImage({ path: img("13_training_time.png"), x: 0.5, y: 3.0, w: 4.2, h: 2.1, sizing: { type: "contain" } });
s17.addImage({ path: img("14_parameter_efficiency.png"), x: 5.1, y: 3.0, w: 4.2, h: 2.1, sizing: { type: "contain" } });

s17.addText("Fig. 6: Training Time Comparison (Left) and Parameter Efficiency (Right)", {
  x: 0.5, y: 5.15, w: 9, h: 0.2,
  fontSize: 11, fontFace: F.body, color: C.darkGray, italic: true,
  align: "center", valign: "middle"
});
addPageNumber(s17, 17);

// ==========================
// Slide 18: McNemar Test
// ==========================
let s18 = pres.addSlide();
s18.background = { color: C.white };
titleShape(s18, "Experimental Results");
subtitleShape(s18, "Statistical Significance: McNemar Test");

let mcnemarRows = [
  [
    { text: "Comparison", options: { bold: true, fill: { color: C.tableHeader } } },
    { text: "Statistic (x2)", options: { bold: true, fill: { color: C.tableHeader } } },
    { text: "p-value", options: { bold: true, fill: { color: C.tableHeader } } },
    { text: "Significance", options: { bold: true, fill: { color: C.tableHeader } } },
    { text: "Interpretation", options: { bold: true, fill: { color: C.tableHeader } } }
  ],
  ["DistilBERT vs DistilBERT+LoRA", "0.000", "1.000", "Not Significant", "No statistically significant difference at alpha=0.05"]
];
s18.addTable(mcnemarRows, {
  x: 0.5, y: 1.5, w: 9.0, h: 1.0,
  fontFace: F.alt, fontSize: 12,
  border: { pt: 0.5, color: C.tableBorder },
  colW: [2.2, 1.3, 1.2, 1.5, 2.8],
  valign: "middle", align: "center"
});

s18.addText("Interpretation:", {
  x: 0.6, y: 2.8, w: 8.8, h: 0.3,
  fontSize: 16, fontFace: F.body, color: C.brown, bold: true,
  align: "left", valign: "middle"
});
s18.addText("The McNemar test confirms that DistilBERT+LoRA achieves statistically equivalent performance to full DistilBERT fine-tuning (p = 1.0). This validates LoRA as a parameter-efficient alternative, reducing trainable parameters by ~99.1% (0.59M vs 66.96M) without sacrificing accuracy.", {
  x: 0.6, y: 3.15, w: 8.8, h: 1.2,
  fontSize: 14, fontFace: F.body, color: C.black,
  align: "left", valign: "top"
});
addPageNumber(s18, 18);

// ==========================
// Slide 19: Safe Reply Generation
// ==========================
let s19 = pres.addSlide();
s19.background = { color: C.white };
titleShape(s19, "Safe Reply Generation");
subtitleShape(s19, "Evaluation of Generated Replies");

let replyRows = [
  [{ text: "Metric", options: { bold: true, fill: { color: C.tableHeader } } }, { text: "Value", options: { bold: true, fill: { color: C.tableHeader } } }, { text: "Direction", options: { bold: true, fill: { color: C.tableHeader } } }],
  ["Toxicity Score", "0.066", "Lower is Better"],
  ["Empathy Score", "0.1225", "Higher is Better"],
  ["Relevance Score", "0.5846", "Higher is Better"],
  ["Overall Safety Score", "0.5451", "Higher is Better"]
];
s19.addTable(replyRows, {
  x: 0.5, y: 1.3, w: 5.0, h: 1.6,
  fontFace: F.alt, fontSize: 12,
  border: { pt: 0.5, color: C.tableBorder },
  colW: [2.2, 1.2, 1.6],
  valign: "middle", align: "center"
});

s19.addImage({ path: img("15_reply_safety.png"), x: 5.8, y: 1.1, w: 3.8, h: 2.4, sizing: { type: "contain" } });

s19.addText("Approach:", {
  x: 0.5, y: 3.2, w: 8.8, h: 0.3,
  fontSize: 16, fontFace: F.body, color: C.brown, bold: true,
  align: "left", valign: "middle"
});
s19.addText([
  { text: "DistilGPT-2 fine-tuned for empathetic, non-toxic reply generation.", options: { breakLine: true, bullet: true } },
  { text: "Decoding parameters: temperature=0.7, top_p=0.9, max_new_tokens=50.", options: { breakLine: true, bullet: true } },
  { text: "Safety evaluation via Perspective API toxicity scoring and semantic relevance metrics.", options: { breakLine: true, bullet: true } }
], {
  x: 0.5, y: 3.55, w: 8.8, h: 1.2,
  fontFace: F.body, color: C.black, fontSize: 13,
  paraSpaceAfter: 3,
  bullet: { color: C.teal }
});
addPageNumber(s19, 19);

// ==========================
// Slide 20: Conclusion & Future Scope
// ==========================
let s20 = pres.addSlide();
s20.background = { color: C.white };
titleShape(s20, "Conclusion and Future Scope");

s20.addText("Conclusion:", {
  x: 0.6, y: 1.0, w: 8.8, h: 0.3,
  fontSize: 18, fontFace: F.body, color: C.brown, bold: true,
  align: "left", valign: "middle"
});
s20.addText([
  { text: "Achieved 99.74% test accuracy with DistilBERT for 4-class cyberbullying detection.", options: { breakLine: true, bullet: true } },
  { text: "Demonstrated LoRA as a viable parameter-efficient alternative (99.1% parameter reduction) with equivalent performance.", options: { breakLine: true, bullet: true } },
  { text: "Built an integrated safe-reply generator with low toxicity (0.066) and high relevance (0.58).", options: { breakLine: true, bullet: true } },
  { text: "Comprehensive evaluation including confusion matrices, McNemar statistical testing, and efficiency benchmarking.", options: { breakLine: true, bullet: true } }
], {
  x: 0.6, y: 1.35, w: 8.8, h: 1.6,
  fontFace: F.body, color: C.black, fontSize: 14,
  paraSpaceAfter: 3,
  bullet: { color: C.teal }
});

s20.addText("Future Scope:", {
  x: 0.6, y: 3.1, w: 8.8, h: 0.3,
  fontSize: 18, fontFace: F.body, color: C.brown, bold: true,
  align: "left", valign: "middle"
});
s20.addText([
  { text: "Extend to multi-lingual cyberbullying detection.", options: { breakLine: true, bullet: true } },
  { text: "Integrate real-time social media streaming APIs for live moderation.", options: { breakLine: true, bullet: true } },
  { text: "Explore larger LLMs (LLaMA, Mistral) for improved safe-reply quality.", options: { breakLine: true, bullet: true } },
  { text: "Add explainability modules (SHAP, LIME) to interpret model predictions.", options: { breakLine: true, bullet: true } },
  { text: "Deploy as a browser extension or platform plugin.", options: { breakLine: true, bullet: true } }
], {
  x: 0.6, y: 3.45, w: 8.8, h: 1.5,
  fontFace: F.body, color: C.black, fontSize: 14,
  paraSpaceAfter: 3,
  bullet: { color: C.teal }
});
addPageNumber(s20, 20);

// ==========================
// Slide 21: References
// ==========================
let s21 = pres.addSlide();
s21.background = { color: C.white };
titleShape(s21, "References");

let refs = [
  "[1] K. Gutierrez-Batista et al., \"Improving automatic cyberbullying detection by fine-tuning a pre-trained sentence transformer,\" Social Network Analysis and Mining, Springer, vol. 14, no. 1, pp. 1-18, 2024.",
  "[2] S. Muneer et al., \"Cyberbullying detection using stacking ensemble learning and enhanced BERT,\" Information, vol. 14, no. 8, p. 467, 2023.",
  "[3] Z. Yi and A. Zubiaga, \"XP-CB: Cyberbullying detection across social media platforms via platform-aware adversarial encoding,\" in Proc. Int. AAAI Conf. Web Social Media (ICWSM), Atlanta, GA, USA, 2022, pp. 1148-1157.",
  "[4] N. Tabassum and V. Nunavath, \"Hybrid deep learning for multi-class cyberbullying classification using multi-modal data,\" Applied Sciences, vol. 14, no. 24, p. 12007, 2024.",
  "[5] S. Garcia-Mendez and F. De Arriba-Perez, \"Explainable cyberbullying detection using LLMs in stream-based ML,\" in Proc. IEEE Int. Conf. Data Science and Advanced Analytics (DSAA), San Diego, CA, USA, 2024, pp. 1-10.",
  "[6] V. Sanh et al., \"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,\" arXiv preprint arXiv:1910.01108, 2019.",
  "[7] E. J. Hu et al., \"LoRA: Low-rank adaptation of large language models,\" in Proc. Int. Conf. Learn. Representations (ICLR), Virtual, 2022, pp. 1-13.",
  "[8] S. Hoque et al., \"Advancing cyberbullying detection in low-resource languages: A transformer-stacking framework,\" Frontiers in Artificial Intelligence, vol. 8, pp. 1-15, 2025.",
  "[9] B. Mathew et al., \"HateXplain: A benchmark dataset for explainable hate speech detection,\" in Proc. AAAI Conf. Artif. Intell., Virtual, 2021, pp. 14867-14875.",
  "[10] S. Mozafari et al., \"BERT-based transfer learning for hate speech detection in online social media,\" Complex & Intelligent Systems, Springer, vol. 10, no. 2, pp. 2245-2262, 2023.",
  "[11] J. Smith et al., \"Beyond binary: Towards embracing complexities in cyberbullying detection and intervention,\" arXiv preprint arXiv:2409.14285, 2024.",
  "[12] S. Salawu et al., \"Computational approaches to cyberbullying detection: A survey,\" Expert Systems with Applications, vol. 194, p. 116475, 2022.",
  "[13] A. Vaswani et al., \"Attention is all you need,\" in Proc. 31st Int. Conf. Neural Inf. Process. Syst. (NeurIPS), Long Beach, CA, USA, 2017, pp. 5998-6008."
];

let refText = refs.map(r => ({ text: r, options: { breakLine: true } }));
s21.addText(refText, {
  x: 0.5, y: 0.9, w: 9.0, h: 4.3,
  fontFace: F.body, color: C.black, fontSize: 11,
  paraSpaceAfter: 5
});
addPageNumber(s21, 21);

// ==========================
// Write file
// ==========================
const outPath = path.join(ROOT, "FYRP_End-term_Presentation_NEW.pptx");
pres.writeFile({ fileName: outPath })
  .then(() => {
    console.log("Presentation created:", outPath);
    // Rename to final name
    const finalPath = path.join(ROOT, "FYRP_End-term_Presentation.pptx");
    try {
      fs.unlinkSync(finalPath);
    } catch (e) {}
    fs.renameSync(outPath, finalPath);
    console.log("Renamed to:", finalPath);
  })
  .catch(err => { console.error("Error:", err); process.exit(1); });
