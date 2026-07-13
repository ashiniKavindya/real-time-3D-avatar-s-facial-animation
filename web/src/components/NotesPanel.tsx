import { useCallback, useEffect, useState } from 'react';
import { addNote, deleteNote, fetchNotes, type NoteRecord } from '../lib/notesClient';

export function NotesPanel() {
  const [notes, setNotes] = useState<NoteRecord[]>([]);
  const [draft, setDraft] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) fetchNotes().then(setNotes).catch(() => {});
  }, [isOpen]);

  const handleAdd = useCallback(async () => {
    const text = draft.trim();
    if (!text || isSaving) return;
    setIsSaving(true);
    setError(null);
    try {
      setNotes(await addNote(text));
      setDraft('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save note.');
    } finally {
      setIsSaving(false);
    }
  }, [draft, isSaving]);

  const handleDelete = useCallback(async (id: number) => {
    setNotes(await deleteNote(id));
  }, []);

  return (
    <div className="notes-panel">
      <button onClick={() => setIsOpen((v) => !v)} className="notes-toggle-button">
        {isOpen ? 'Hide personal notes' : 'Manage personal notes'}
      </button>

      {isOpen && (
        <div className="notes-body">
          <p className="notes-hint">
            Notes the chatbot can reference when relevant (e.g. hobbies, work, things about you).
          </p>
          <div className="notes-input-row">
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="e.g. I'm learning to play guitar..."
              className="notes-input"
              rows={2}
            />
            <button onClick={() => void handleAdd()} disabled={isSaving || !draft.trim()} className="notes-add-button">
              Add
            </button>
          </div>
          {error && <p className="chat-error">{error}</p>}

          <ul className="notes-list">
            {notes.map((note) => (
              <li key={note.id} className="notes-list-item">
                <span>{note.content}</span>
                <button onClick={() => void handleDelete(note.id)} className="notes-delete-button">
                  Delete
                </button>
              </li>
            ))}
            {notes.length === 0 && <li className="notes-empty">No notes yet.</li>}
          </ul>
        </div>
      )}
    </div>
  );
}
