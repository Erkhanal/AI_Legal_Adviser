
export type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: number;
  rating?: number;
  comment?: string;
};

export type Conversation = {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
};
