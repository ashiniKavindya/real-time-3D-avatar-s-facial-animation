import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';

// Add a case here (and its LangChain integration package) to support another
// provider - everything else in the app talks to whatever this returns, so
// switching providers never requires touching chat/greeting/summary logic.
const GEMINI_MODEL = 'gemini-flash-latest';

export function createChatModel(): BaseChatModel | null {
  const provider = process.env.CHAT_PROVIDER ?? 'gemini';

  switch (provider) {
    case 'gemini': {
      const apiKey = process.env.GEMINI_API_KEY;
      return apiKey ? new ChatGoogleGenerativeAI({ model: GEMINI_MODEL, apiKey }) : null;
    }
    default:
      throw new Error(`Unknown CHAT_PROVIDER: "${provider}". Supported: gemini.`);
  }
}
