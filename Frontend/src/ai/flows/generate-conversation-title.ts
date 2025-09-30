'use server';

/**
 * @fileOverview An AI agent that generates conversation titles based on conversation history.
 *
 * - generateConversationTitle - A function that handles the generation of conversation titles.
 * - GenerateConversationTitleInput - The input type for the generateConversationTitle function.
 * - GenerateConversationTitleOutput - The return type for the generateConversationTitle function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const GenerateConversationTitleInputSchema = z.object({
  conversationHistory: z
    .string()
    .describe('The complete history of the conversation.'),
});
export type GenerateConversationTitleInput = z.infer<
  typeof GenerateConversationTitleInputSchema
>;

const GenerateConversationTitleOutputSchema = z.object({
  title: z.string().describe('The generated title for the conversation.'),
});
export type GenerateConversationTitleOutput = z.infer<
  typeof GenerateConversationTitleOutputSchema
>;

export async function generateConversationTitle(
  input: GenerateConversationTitleInput
): Promise<GenerateConversationTitleOutput> {
  return generateConversationTitleFlow(input);
}

const prompt = ai.definePrompt({
  name: 'generateConversationTitlePrompt',
  input: {schema: GenerateConversationTitleInputSchema},
  output: {schema: GenerateConversationTitleOutputSchema},
  prompt: `You are an AI assistant that suggests conversation titles based on the conversation history.  The title should be descriptive and concise.

Conversation History: {{{conversationHistory}}}

Suggest a title for this conversation:
`,
});

const generateConversationTitleFlow = ai.defineFlow(
  {
    name: 'generateConversationTitleFlow',
    inputSchema: GenerateConversationTitleInputSchema,
    outputSchema: GenerateConversationTitleOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
