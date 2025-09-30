import { db } from './firebase';
import { collection, addDoc, getDocs, doc, updateDoc, deleteDoc, query, orderBy, setDoc } from 'firebase/firestore';
import type { Conversation, Message } from './types';

if (!db) {
  console.warn("Firestore is not initialized. Make sure your Firebase credentials are set up correctly in .env.local. All Firestore operations will fail.");
}

// Helper to get the conversations collection for a user
const getConversationsCollection = (userId: string) => {
  if (!db) throw new Error("Firestore not initialized");
  return collection(db, 'users', userId, 'conversations');
};

// Get all conversations for a user
export const getConversations = async (userId: string): Promise<Conversation[]> => {
  const conversationsCollection = getConversationsCollection(userId);
  const q = query(conversationsCollection, orderBy('createdAt', 'desc'));
  const snapshot = await getDocs(q);
  return snapshot.docs.map(doc => {
    const conversation = { id: doc.id, ...doc.data() } as Conversation;
    // Ensure messages have default values for optional fields to avoid sending `undefined` back to Firestore.
    conversation.messages = (conversation.messages || []).map(m => ({
      ...m,
      rating: m.rating || 0,
      comment: m.comment || ''
    }));
    return conversation;
  });
};

// Add a new conversation with a specific ID
export const addConversationWithId = async (userId: string, conversation: Conversation): Promise<void> => {
    if (!db) throw new Error("Firestore not initialized");
    const conversationDoc = doc(db, 'users', userId, 'conversations', conversation.id);
    const { id, ...conversationData } = conversation;
    await setDoc(conversationDoc, conversationData);
};

// Update an existing conversation
export const updateConversation = async (userId: string, conversationId: string, updates: Partial<Omit<Conversation, 'id'>>): Promise<void> => {
  if (!db) throw new Error("Firestore not initialized");
  const conversationDoc = doc(db, 'users', userId, 'conversations', conversationId);
  await updateDoc(conversationDoc, updates);
};

// Delete a conversation
export const deleteConversation = async (userId: string, conversationId: string): Promise<void> => {
  if (!db) throw new Error("Firestore not initialized");
  const conversationDoc = doc(db, 'users', userId, 'conversations', conversationId);
  await deleteDoc(conversationDoc);
};
