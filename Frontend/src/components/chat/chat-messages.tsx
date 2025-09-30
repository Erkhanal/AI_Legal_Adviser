
"use client";

import * as React from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { type Message } from "@/lib/types";
import { ChatMessage } from "./chat-message";
import { Gavel, RefreshCw } from "lucide-react";
import { ThinkingStep } from "./thinking-step";
import { type ThinkingStep as ThinkingStepType } from "@/lib/api-types";
import { format, isSameDay, isToday, isYesterday } from "date-fns";
import { Button } from "@/components/ui/button";
import type { User } from 'firebase/auth';
import { useTranslation } from "../language-provider";

interface ChatMessagesProps {
  user: User | null;
  messages: Message[];
  conversationCreatedAt: number;
  onRateMessage: (messageId: string, rating: number) => void;
  onCommentMessage: (messageId: string, comment: string) => void;
  onEditMessage: (message: Message) => void;
  onRegenerateResponse: () => void;
  isGenerating: boolean;
  thinkingSteps: (ThinkingStepType & { duration?: string })[];
  conversationId: string | null;
}

export function ChatMessages({ user, messages, conversationCreatedAt, onRateMessage, onCommentMessage, onEditMessage, onRegenerateResponse, isGenerating, thinkingSteps, conversationId }: ChatMessagesProps) {
  const scrollAreaRef = React.useRef<HTMLDivElement>(null);
  const viewportRef = React.useRef<HTMLDivElement>(null);
  const prevConversationId = React.useRef<string | null>(null);
  const prevMessageLength = React.useRef(0);
  const { t, dateLocale } = useTranslation();
  let lastDate: Date | null = null;

  React.useEffect(() => {
    if (!viewportRef.current) return;
    
    if (prevConversationId.current !== conversationId) {
      viewportRef.current.scrollTop = 0;
      prevConversationId.current = conversationId;
    } 
    else if (messages.length > prevMessageLength.current || isGenerating) {
      viewportRef.current.scrollTop = viewportRef.current.scrollHeight;
    }
    
    prevMessageLength.current = messages.length;
  }, [messages, isGenerating, thinkingSteps, conversationId]);

  const lastMessage = messages[messages.length - 1];
  const showRegenerateButton = !isGenerating && lastMessage && lastMessage.role === 'assistant';

  return (
    <ScrollArea className="flex-1" ref={scrollAreaRef} viewportRef={viewportRef}>
      <div className="p-4 space-y-2">
        {messages.map((message, index) => {
          const messageTimestamp = message.createdAt || conversationCreatedAt + index;
          const currentDate = new Date(messageTimestamp);
          let dateSeparator: React.ReactNode = null;
          const isLastMessage = index === messages.length - 1;

          if (!lastDate || !isSameDay(currentDate, lastDate)) {
            let dateLabel;
            if (isToday(currentDate)) {
              dateLabel = t('today');
            } else if (isYesterday(currentDate)) {
              dateLabel = t('yesterday');
            } else {
              dateLabel = format(currentDate, 'MMMM d, yyyy', { locale: dateLocale });
            }
            dateSeparator = (
              <div className="text-center text-xs text-muted-foreground my-4">
                {dateLabel}
              </div>
            );
            lastDate = currentDate;
          }

          return (
            <React.Fragment key={message.id}>
              {dateSeparator}
              <ChatMessage 
                user={user}
                message={{ ...message, createdAt: messageTimestamp }}
                isLastMessage={isLastMessage}
                isGenerating={isGenerating}
                onRateMessage={onRateMessage} 
                onCommentMessage={onCommentMessage}
                onEditMessage={onEditMessage}
                onRegenerateResponse={onRegenerateResponse}
              />
            </React.Fragment>
          )
        })}
        
        {isGenerating && thinkingSteps.length > 0 && (
          <div className="flex items-start gap-2 py-2 px-4">
            <span className="w-8 h-8 flex items-center justify-center text-2xl shrink-0">
              <Gavel />
            </span>
            <div className="flex flex-col items-start gap-1 pt-1">
              {thinkingSteps.map((step, index) => (
                <ThinkingStep key={index} step={step} />
              ))}
            </div>
          </div>
        )}

        {isGenerating && thinkingSteps.length === 0 && messages[messages.length-1]?.role === 'user' && (
           <div className="flex items-start gap-4 p-4">
            <Gavel className="w-8 h-8 text-muted-foreground" />
             <div className="flex items-center gap-2 pt-2">
                <span className="h-2 w-2 bg-muted-foreground rounded-full animate-pulse [animation-delay:-0.3s]"></span>
                <span className="h-2 w-2 bg-muted-foreground rounded-full animate-pulse [animation-delay:-0.15s]"></span>
                <span className="h-2 w-2 bg-muted-foreground rounded-full animate-pulse"></span>
            </div>
          </div>
        )}

         {showRegenerateButton && (
          <div className="flex justify-center py-4">
            <Button variant="outline" size="sm" onClick={onRegenerateResponse}>
              <RefreshCw className="mr-2 h-4 w-4" />
              {t('regenerateResponse')}
            </Button>
          </div>
        )}
      </div>
    </ScrollArea>
  );
}
