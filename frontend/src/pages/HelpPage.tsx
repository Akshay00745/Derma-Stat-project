import React, { useState } from 'react';
import { ChevronDown, Mail, HelpCircle, ExternalLink } from 'lucide-react';

interface FAQItem {
  question: string;
  answer: string;
}

export const HelpPage: React.FC = () => {
  const [openIdx, setOpenIdx] = useState<number | null>(0);

  const faqs: FAQItem[] = [
    {
      question: 'How do I upload an image?',
      answer:
        'Navigate to the "Upload Image" page from the navigation sidebar. You can either drag and drop a dermatoscopic image directly onto the drop zone or click "Choose File" to browse your local device. Once loaded, click "Run AI Analysis" to begin inference.',
    },
    {
      question: 'What image formats are supported?',
      answer:
        'Dermastat AI Vision natively supports JPG, JPEG, and PNG dermoscopic photographs. For optimal diagnostic results, high-resolution photographs with clean lighting and centered lesion boundaries are recommended (minimum 224 × 224 pixels).',
    },
    {
      question: 'What do the seven categories mean?',
      answer:
        'The 7 diagnostic categories originate from the HAM10000 benchmark: AKIEC (Actinic Keratoses / Bowen\'s disease), BCC (Basal Cell Carcinoma), BKL (Benign Keratosis-like Lesions), DF (Dermatofibroma), MEL (Melanoma), NV (Melanocytic Nevi / common moles), and VASC (Vascular Lesions).',
    },
    {
      question: 'How is the prediction confidence calculated?',
      answer:
        'The model produces raw logits from its final classification layer, which are passed through the mathematical Softmax function to produce a normalized probability distribution across all 7 categories summing to 100%. The confidence score represents the model\'s highest assigned class probability.',
    },
    {
      question: 'What does the uncertainty warning mean?',
      answer:
        'Using validation-set calibration, our safety engine established an optimal threshold of 83.15%. When the model\'s top confidence falls below 83.15%, an uncertainty flag is raised, indicating that the lesion displays visual ambiguity between closely related diagnostic classes and requires prioritized clinical review.',
    },
    {
      question: 'How is Macro F1-score different from accuracy?',
      answer:
        'Accuracy measures the total percentage of correct predictions over the entire dataset, which can be misleadingly inflated when common classes (like Melanocytic Nevi) dominate 67% of cases. The Macro F1-score computes the harmonic mean of precision and recall for each of the 7 classes independently before taking their unweighted average, giving equal priority to rare, critical malignancies like Melanoma and Basal Cell Carcinoma.',
    },
    {
      question: 'Is this tool a medical diagnostic system?',
      answer:
        'No. Dermastat AI Vision is an investigational and educational AI prototype. It is not approved by the FDA, CE, or any regulatory medical authority. It must never be used as a standalone diagnostic tool or to replace clinical examination, biopsy, or dermatological consultation.',
    },
  ];

  const toggleFAQ = (index: number) => {
    setOpenIdx(openIdx === index ? null : index);
  };

  return (
    <div className="p-8 max-w-4xl mx-auto space-y-8">
      {/* Title */}
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Help & Support</h2>
        <p className="text-sm text-slate-500 mt-1">
          Find answers to common questions about using Dermastat AI Vision.
        </p>
      </div>

      {/* 7 Accordion FAQs (Panel 8 core) */}
      <div className="bg-white rounded-3xl p-6 border border-slate-200/80 shadow-xs divide-y divide-slate-100">
        {faqs.map((faq, idx) => {
          const isOpen = openIdx === idx;
          return (
            <div key={idx} className="py-4 first:pt-1 last:pb-1">
              <button
                onClick={() => toggleFAQ(idx)}
                className="w-full flex items-center justify-between text-left text-sm font-bold text-slate-800 hover:text-blue-600 transition-colors group"
              >
                <span>{faq.question}</span>
                <ChevronDown
                  className={`w-4 h-4 text-slate-400 group-hover:text-blue-600 transition-transform duration-200 shrink-0 ml-4 ${
                    isOpen ? 'rotate-180 text-blue-600' : ''
                  }`}
                />
              </button>

              {isOpen && (
                <div className="mt-3 text-xs text-slate-600 leading-relaxed pr-6 animate-fadeIn">
                  {faq.answer}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Contact Channel Footer (Panel 8 bottom) */}
      <div className="text-center pt-2">
        <p className="text-xs text-slate-500 font-medium">
          Still have questions? Contact us at{' '}
          <a
            href="mailto:support@dermastat.ai"
            className="text-blue-600 hover:underline font-semibold"
          >
            support@dermastat.ai
          </a>
        </p>
      </div>
    </div>
  );
};
