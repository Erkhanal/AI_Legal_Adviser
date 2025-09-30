
"use client";

import * as React from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Send, Square } from "lucide-react";
import { useTranslation } from "../language-provider";

interface ChatInputProps {
  value: string;
  onValueChange: (value: string) => void;
  onSendMessage: (content: string) => void;
  isGenerating: boolean;
  onStopGenerating: () => void;
}

export const ChatInput = React.forwardRef<HTMLTextAreaElement, ChatInputProps>(
  ({ value, onValueChange, onSendMessage, isGenerating, onStopGenerating }, ref) => {
    const { t } = useTranslation();

    const handleSendMessageClick = () => {
      if (value.trim() && !isGenerating) {
        onSendMessage(value.trim());
      }
    };

    const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === "Enter" && !event.altKey) {
        event.preventDefault();
        handleSendMessageClick();
      }
    };

    React.useEffect(() => {
        const textarea = (ref as React.RefObject<HTMLTextAreaElement>)?.current;
        if (textarea) {
          textarea.style.height = 'auto';
          const scrollHeight = textarea.scrollHeight;
          const maxHeight = parseInt(getComputedStyle(textarea).maxHeight, 10);
          textarea.style.height = `${Math.min(scrollHeight, maxHeight)}px`;
        }
    }, [value, ref]);

    return (
      <div className="p-4 bg-background border-t">
        <div className="flex items-center w-full p-1 pl-4 border rounded-full bg-muted/50">
          <Textarea
            ref={ref}
            value={value}
            onChange={(e) => onValueChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={t('typeYourMessage')}
            className="flex-1 py-2 resize-none max-h-48 bg-transparent border-none focus-visible:ring-0 focus-visible:ring-offset-0"
            rows={1}
            disabled={isGenerating}
            aria-label="Chat input"
          />
          {isGenerating ? (
            <Button
              type="button"
              size="icon"
              variant="destructive"
              className="shrink-0 rounded-full"
              onClick={onStopGenerating}
              aria-label={t('stopGenerating')}
            >
              <Square className="h-4 w-4" />
            </Button>
          ) : (
            <Button
              type="submit"
              size="icon"
              className="shrink-0 rounded-full"
              onClick={handleSendMessageClick}
              disabled={!value.trim() || isGenerating}
              aria-label={t('sendMessage')}
            >
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    );
  }
);

ChatInput.displayName = 'ChatInput';
