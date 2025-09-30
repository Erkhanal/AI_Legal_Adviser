
"use client";

import * as React from "react";
import {
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarMenuAction,
  SidebarSeparator,
} from "@/components/ui/sidebar";
import {
  MessageSquare,
  Trash2,
  Edit,
  Check,
  X,
  PlusCircle,
  Search,
} from "lucide-react";
import { type Conversation } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { cn } from "@/lib/utils";
import { useTranslation } from "../language-provider";

interface ConversationListProps {
  conversations: Conversation[];
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onRenameConversation: (id: string, newTitle: string) => void;
  onNewConversation: () => void;
}

export function ConversationList({
  conversations,
  activeConversationId,
  onSelectConversation,
  onDeleteConversation,
  onRenameConversation,
  onNewConversation,
}: ConversationListProps) {
  const [editingId, setEditingId] = React.useState<string | null>(null);
  const [renameValue, setRenameValue] = React.useState("");
  const [searchTerm, setSearchTerm] = React.useState("");
  const { t } = useTranslation();

  const handleEdit = (conversation: Conversation) => {
    setEditingId(conversation.id);
    setRenameValue(conversation.title);
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setRenameValue("");
  };

  const handleSaveRename = (id: string) => {
    if (renameValue.trim()) {
      onRenameConversation(id, renameValue.trim());
      handleCancelEdit();
    }
  };

  const filteredConversations = React.useMemo(() =>
    conversations.filter((conversation) =>
      conversation.title.toLowerCase().includes(searchTerm.toLowerCase())
    ),
    [conversations, searchTerm]
  );

  return (
    <div className="flex flex-col h-full">
      <div className="p-2">
        <Button
          variant="outline"
          className="w-full justify-start gap-2 group-data-[collapsible=icon]:size-8 group-data-[collapsible=icon]:justify-center"
          onClick={onNewConversation}
        >
          <PlusCircle className="size-4" />
          <span className="group-data-[collapsible=icon]:hidden">{t('newChat')}</span>
        </Button>
      </div>
      <div className="px-2 pb-2">
        <div className="relative group-data-[collapsible=icon]:hidden">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder={t('searchChats')}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9"
          />
        </div>
      </div>
      <SidebarSeparator />
      {filteredConversations.length > 0 && (
        <div className="px-2 my-2 text-xs font-semibold tracking-wider uppercase text-muted-foreground group-data-[collapsible=icon]:hidden">
            {t('chatHistory')}
        </div>
      )}
      <SidebarMenu className="flex-1 p-2 pt-0">
        {filteredConversations.map((conversation) => (
          <SidebarMenuItem key={conversation.id}>
            {editingId === conversation.id ? (
              <div className="flex items-center gap-1 p-1">
                <Input
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSaveRename(conversation.id)}
                  className="h-8"
                  autoFocus
                />
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => handleSaveRename(conversation.id)}>
                  <Check className="h-4 w-4" />
                </Button>
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={handleCancelEdit}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ) : (
              <SidebarMenuButton
                onClick={() => onSelectConversation(conversation.id)}
                isActive={conversation.id === activeConversationId}
                className="w-full justify-start"
                tooltip={conversation.title}
              >
                <MessageSquare />
                <span className="truncate flex-1 text-left">{conversation.title}</span>
              </SidebarMenuButton>
            )}

            {editingId !== conversation.id && (
              <>
                <SidebarMenuAction
                  onClick={() => handleEdit(conversation)}
                  aria-label="Rename conversation"
                  showOnHover
                >
                  <Edit />
                </SidebarMenuAction>
                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <SidebarMenuAction
                      aria-label="Delete conversation"
                      className={cn(
                        "right-8 peer-hover/menu-button:text-destructive hover:!text-destructive",
                        "group-focus-within/menu-item:opacity-100 group-hover/menu-item:opacity-100 data-[state=open]:opacity-100 peer-data-[active=true]/menu-button:text-sidebar-accent-foreground md:opacity-0"
                      )}
                    >
                      <Trash2 />
                    </SidebarMenuAction>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>{t('areYouSure')}</AlertDialogTitle>
                      <AlertDialogDescription>
                        {t('deleteConversationConfirmation', { title: conversation.title })}
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>{t('cancel')}</AlertDialogCancel>
                      <AlertDialogAction onClick={() => onDeleteConversation(conversation.id)}>
                        {t('delete')}
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </>
            )}
          </SidebarMenuItem>
        ))}
      </SidebarMenu>
    </div>
  );
}
