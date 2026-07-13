import { db } from './db.js';

const DEFAULT_BOT_NAME = 'Friend';

export function getBotName(userId: string): string {
  const row = db.prepare('SELECT bot_name FROM user_settings WHERE user_id = ?').get(userId);
  return row ? String(row.bot_name) : DEFAULT_BOT_NAME;
}

export function setBotName(userId: string, botName: string): void {
  db.prepare(
    `INSERT INTO user_settings (user_id, bot_name) VALUES (?, ?)
     ON CONFLICT(user_id) DO UPDATE SET bot_name = excluded.bot_name`,
  ).run(userId, botName);
}
