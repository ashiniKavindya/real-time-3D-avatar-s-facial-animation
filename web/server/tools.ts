import { tool } from '@langchain/core/tools';
import { z } from 'zod';

const getCurrentDateTime = tool(
  async () => new Date().toLocaleString('en-US', { dateStyle: 'full', timeStyle: 'short' }),
  {
    name: 'get_current_datetime',
    description: 'Get the current date and time. Use this whenever the user asks what day, date, or time it is.',
    schema: z.object({}),
  },
);

export const chatTools = [getCurrentDateTime];
