
"use client";

import * as React from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Gavel, User, Star, MessageSquarePlus, MessageSquareText, RefreshCw, Pencil } from "lucide-react";
import { type Message } from "@/lib/types";
import { cn } from "@/lib/utils";
import { CodeBlock } from "@/components/code-block";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Textarea } from "@/components/ui/textarea";
import { format } from "date-fns";
import type { User as FirebaseAuthUser } from 'firebase/auth';
import { useTranslation } from "../language-provider";

interface ChatMessageProps {
  user: FirebaseAuthUser | null;
  message: Message;
  isLastMessage: boolean;
  isGenerating: boolean;
  onRateMessage: (messageId: string, rating: number) => void;
  onCommentMessage: (messageId:string, comment: string) => void;
  onEditMessage: (message: Message) => void;
  onRegenerateResponse: () => void;
}

function ChatMessageActions({ onEdit, onRegenerate }: { onEdit: () => void, onRegenerate: () => void }) {
    const { t } = useTranslation();
    return (
        <div className="flex items-center gap-1">
            <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground" onClick={onEdit}>
                <Pencil className="h-4 w-4" />
                <span className="sr-only">{t('editMessage')}</span>
            </Button>
            <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground" onClick={onRegenerate}>
                <RefreshCw className="h-4 w-4" />
                <span className="sr-only">{t('regenerateResponse')}</span>
            </Button>
        </div>
    );
}


export function ChatMessage({ user, message, isLastMessage, isGenerating, onRateMessage, onCommentMessage, onEditMessage, onRegenerateResponse }: ChatMessageProps) {
  const { role, content, rating, comment, createdAt } = message;
  const { t, dateLocale } = useTranslation();
  const [hoverRating, setHoverRating] = React.useState(0);
  const [commentText, setCommentText] = React.useState(comment || "");
  const [isPopoverOpen, setIsPopoverOpen] = React.useState(false);

  const handleSaveComment = () => {
    onCommentMessage(message.id, commentText);
    setIsPopoverOpen(false);
  };
  
  React.useEffect(() => {
    setCommentText(comment || "");
  }, [comment]);

  const isUser = role === 'user';
  const showUserActions = isUser && isLastMessage && !isGenerating;

  return (
    <div className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}>
        <div className={cn("flex gap-4", isUser ? "flex-row-reverse items-end" : "flex-row items-end")}>
            <Avatar className={cn("w-8 h-8 border", isUser && "mb-3")}>
                {isUser ? (
                    user?.photoURL ? (
                        <AvatarImage src={user.photoURL} alt={user.displayName || 'User avatar'} />
                    ) : (
                        <AvatarFallback><User /></AvatarFallback>
                    )
                ) : (
                    <AvatarFallback><Gavel /></AvatarFallback>
                )}
            </Avatar>
            <div className={cn("group max-w-prose flex flex-col", isUser ? "items-end" : "items-start")}>
                <div className={cn("flex items-center", isUser ? "flex-row-reverse" : "flex-row")}>
                    <div className={cn(
                        "relative rounded-lg border bg-card text-card-foreground shadow-sm",
                        isUser ? "rounded-br-none" : "rounded-bl-none"
                    )}>
                        <div className="p-4 space-y-4 break-words text-xs">
                        <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                                p: ({...props}) => <p className="whitespace-pre-wrap" {...props} />,
                                code({node, inline, className, children, ...props}) {
                                    const match = /language-(\w+)/.exec(className || '');
                                    if (!inline && match) {
                                    return <CodeBlock language={match[1]} code={String(children).replace(/\n$/, '')} />
                                    }
                                    return (
                                    <code className={cn("font-code bg-black/10 dark:bg-white/10 px-1 py-0.5 rounded-sm", className)} {...props}>
                                        {children}
                                    </code>
                                    );
                                },
                            }}
                            >
                            {content}
                        </ReactMarkdown>
                        </div>
                        {!isUser && content && (
                            <div className="flex items-center justify-between border-t px-4 py-2">
                                <div className="flex items-center gap-0.5" onMouseLeave={() => setHoverRating(0)}>
                                    {[1, 2, 3, 4, 5].map((star) => (
                                    <Button
                                        key={star}
                                        variant="ghost"
                                        size="icon"
                                        className={cn("h-7 w-7", (hoverRating >= star || (!hoverRating && rating && rating >= star)) ? 'text-primary' : 'text-muted-foreground')}
                                        onMouseEnter={() => setHoverRating(star)}
                                        onClick={() => onRateMessage(message.id, rating === star ? 0 : star)}
                                        aria-label={t('rateStar', { star })}
                                    >
                                        <Star className="h-4 w-4" fill="currentColor" />
                                    </Button>
                                    ))}
                                    <Popover open={isPopoverOpen} onOpenChange={setIsPopoverOpen}>
                                    <PopoverTrigger asChild>
                                        <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground" aria-label={comment ? t('viewEditComment') : t('addComment')}>
                                            {comment ? <MessageSquareText className="h-4 w-4 text-primary" /> : <MessageSquarePlus className="h-4 w-4" />}
                                        </Button>
                                    </PopoverTrigger>
                                    <PopoverContent className="w-80">
                                        <div className="grid gap-4">
                                            <div className="space-y-2">
                                                <h4 className="font-medium leading-none">{t('comment')}</h4>
                                                <p className="text-xs text-muted-foreground">{t('addEditCommentDescription')}</p>
                                            </div>
                                            <div className="grid gap-2">
                                                <Textarea value={commentText} onChange={(e) => setCommentText(e.target.value)} placeholder={t('yourCommentPlaceholder')} rows={3} />
                                                <Button onClick={handleSaveComment}>{t('save')}</Button>
                                            </div>
                                        </div>
                                    </PopoverContent>
                                    </Popover>
                                </div>
                                <span className="text-xs text-muted-foreground">{format(new Date(createdAt), 'p', { locale: dateLocale })}</span>
                            </div>
                        )}
                    </div>
                </div>
                 {showUserActions && (
                    <div className="mt-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <ChatMessageActions
                            onEdit={() => onEditMessage(message)}
                            onRegenerate={() => onRegenerateResponse()}
                        />
                    </div>
                )}
                {isUser && !showUserActions && <span className="text-xs text-muted-foreground mt-1">{format(new Date(createdAt), 'p', { locale: dateLocale })}</span>}
            </div>
        </div>
    </div>
  );
}
