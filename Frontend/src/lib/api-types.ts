export interface ThinkingStep {
  step: 'Understanding' | 'Searching' | 'Answering' | string;
  status: 'processing' | 'result';
  message: string;
  icon: {
    emoji: string;
  };
}

export interface AnswerChunk {
  content: string;
}
