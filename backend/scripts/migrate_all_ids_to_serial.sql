-- Migrate all business table primary keys from UUID to auto-incrementing integer keys.
-- Run this manually against an existing UUID-based database before starting the updated app.
-- New databases do not need this script; SQLAlchemy will create integer identity keys.

BEGIN;

LOCK TABLE document_chunks IN ACCESS EXCLUSIVE MODE;
LOCK TABLE messages IN ACCESS EXCLUSIVE MODE;
LOCK TABLE documents IN ACCESS EXCLUSIVE MODE;
LOCK TABLE sessions IN ACCESS EXCLUSIVE MODE;
LOCK TABLE knowledge_bases IN ACCESS EXCLUSIVE MODE;

ALTER TABLE knowledge_bases ADD COLUMN id_int integer;
ALTER TABLE sessions ADD COLUMN id_int integer;
ALTER TABLE documents ADD COLUMN id_int integer;
ALTER TABLE messages ADD COLUMN id_int integer;
ALTER TABLE document_chunks ADD COLUMN id_int integer;

CREATE SEQUENCE IF NOT EXISTS knowledge_bases_id_seq AS integer;
CREATE SEQUENCE IF NOT EXISTS sessions_id_seq AS integer;
CREATE SEQUENCE IF NOT EXISTS documents_id_seq AS integer;
CREATE SEQUENCE IF NOT EXISTS messages_id_seq AS integer;
CREATE SEQUENCE IF NOT EXISTS document_chunks_id_seq AS integer;

UPDATE knowledge_bases SET id_int = nextval('knowledge_bases_id_seq');
UPDATE sessions SET id_int = nextval('sessions_id_seq');
UPDATE documents SET id_int = nextval('documents_id_seq');
UPDATE messages SET id_int = nextval('messages_id_seq');
UPDATE document_chunks SET id_int = nextval('document_chunks_id_seq');

ALTER TABLE knowledge_bases ALTER COLUMN id_int SET NOT NULL;
ALTER TABLE sessions ALTER COLUMN id_int SET NOT NULL;
ALTER TABLE documents ALTER COLUMN id_int SET NOT NULL;
ALTER TABLE messages ALTER COLUMN id_int SET NOT NULL;
ALTER TABLE document_chunks ALTER COLUMN id_int SET NOT NULL;

ALTER TABLE sessions ADD COLUMN knowledge_base_id_int integer;
ALTER TABLE documents ADD COLUMN knowledge_base_id_int integer;
ALTER TABLE messages ADD COLUMN session_id_int integer;
ALTER TABLE document_chunks ADD COLUMN document_id_int integer;

UPDATE sessions s
SET knowledge_base_id_int = kb.id_int
FROM knowledge_bases kb
WHERE s.knowledge_base_id = kb.id;

UPDATE documents d
SET knowledge_base_id_int = kb.id_int
FROM knowledge_bases kb
WHERE d.knowledge_base_id = kb.id;

UPDATE messages m
SET session_id_int = s.id_int
FROM sessions s
WHERE m.session_id = s.id;

UPDATE document_chunks c
SET document_id_int = d.id_int
FROM documents d
WHERE c.document_id = d.id;

ALTER TABLE documents ALTER COLUMN knowledge_base_id_int SET NOT NULL;
ALTER TABLE messages ALTER COLUMN session_id_int SET NOT NULL;
ALTER TABLE document_chunks ALTER COLUMN document_id_int SET NOT NULL;

DO $$
DECLARE
    r record;
BEGIN
    FOR r IN
        SELECT conrelid::regclass AS table_name, conname
        FROM pg_constraint
        WHERE contype = 'f'
          AND conrelid IN (
              'sessions'::regclass,
              'documents'::regclass,
              'messages'::regclass,
              'document_chunks'::regclass
          )
          AND confrelid IN (
              'knowledge_bases'::regclass,
              'sessions'::regclass,
              'documents'::regclass
          )
    LOOP
        EXECUTE format('ALTER TABLE %s DROP CONSTRAINT %I', r.table_name, r.conname);
    END LOOP;
END $$;

ALTER TABLE document_chunks DROP CONSTRAINT document_chunks_pkey;
ALTER TABLE messages DROP CONSTRAINT messages_pkey;
ALTER TABLE documents DROP CONSTRAINT documents_pkey;
ALTER TABLE sessions DROP CONSTRAINT sessions_pkey;
ALTER TABLE knowledge_bases DROP CONSTRAINT knowledge_bases_pkey;

ALTER TABLE sessions DROP COLUMN knowledge_base_id;
ALTER TABLE documents DROP COLUMN knowledge_base_id;
ALTER TABLE messages DROP COLUMN session_id;
ALTER TABLE document_chunks DROP COLUMN document_id;

ALTER TABLE knowledge_bases DROP COLUMN id;
ALTER TABLE sessions DROP COLUMN id;
ALTER TABLE documents DROP COLUMN id;
ALTER TABLE messages DROP COLUMN id;
ALTER TABLE document_chunks DROP COLUMN id;

ALTER TABLE knowledge_bases RENAME COLUMN id_int TO id;
ALTER TABLE sessions RENAME COLUMN id_int TO id;
ALTER TABLE documents RENAME COLUMN id_int TO id;
ALTER TABLE messages RENAME COLUMN id_int TO id;
ALTER TABLE document_chunks RENAME COLUMN id_int TO id;

ALTER TABLE sessions RENAME COLUMN knowledge_base_id_int TO knowledge_base_id;
ALTER TABLE documents RENAME COLUMN knowledge_base_id_int TO knowledge_base_id;
ALTER TABLE messages RENAME COLUMN session_id_int TO session_id;
ALTER TABLE document_chunks RENAME COLUMN document_id_int TO document_id;

ALTER SEQUENCE knowledge_bases_id_seq OWNED BY knowledge_bases.id;
ALTER SEQUENCE sessions_id_seq OWNED BY sessions.id;
ALTER SEQUENCE documents_id_seq OWNED BY documents.id;
ALTER SEQUENCE messages_id_seq OWNED BY messages.id;
ALTER SEQUENCE document_chunks_id_seq OWNED BY document_chunks.id;

ALTER TABLE knowledge_bases ALTER COLUMN id SET DEFAULT nextval('knowledge_bases_id_seq');
ALTER TABLE sessions ALTER COLUMN id SET DEFAULT nextval('sessions_id_seq');
ALTER TABLE documents ALTER COLUMN id SET DEFAULT nextval('documents_id_seq');
ALTER TABLE messages ALTER COLUMN id SET DEFAULT nextval('messages_id_seq');
ALTER TABLE document_chunks ALTER COLUMN id SET DEFAULT nextval('document_chunks_id_seq');

ALTER TABLE knowledge_bases ADD CONSTRAINT knowledge_bases_pkey PRIMARY KEY (id);
ALTER TABLE sessions ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);
ALTER TABLE documents ADD CONSTRAINT documents_pkey PRIMARY KEY (id);
ALTER TABLE messages ADD CONSTRAINT messages_pkey PRIMARY KEY (id);
ALTER TABLE document_chunks ADD CONSTRAINT document_chunks_pkey PRIMARY KEY (id);

ALTER TABLE sessions
    ADD CONSTRAINT sessions_knowledge_base_id_fkey
    FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases(id) ON DELETE SET NULL;
ALTER TABLE documents
    ADD CONSTRAINT documents_knowledge_base_id_fkey
    FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE;
ALTER TABLE messages
    ADD CONSTRAINT messages_session_id_fkey
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE;
ALTER TABLE document_chunks
    ADD CONSTRAINT document_chunks_document_id_fkey
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS ix_sessions_knowledge_base_id ON sessions(knowledge_base_id);
CREATE INDEX IF NOT EXISTS ix_documents_knowledge_base_id ON documents(knowledge_base_id);
CREATE INDEX IF NOT EXISTS ix_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS ix_document_chunks_document_id ON document_chunks(document_id);

SELECT setval('knowledge_bases_id_seq', GREATEST((SELECT COALESCE(MAX(id), 0) FROM knowledge_bases), 1), (SELECT COUNT(*) > 0 FROM knowledge_bases));
SELECT setval('sessions_id_seq', GREATEST((SELECT COALESCE(MAX(id), 0) FROM sessions), 1), (SELECT COUNT(*) > 0 FROM sessions));
SELECT setval('documents_id_seq', GREATEST((SELECT COALESCE(MAX(id), 0) FROM documents), 1), (SELECT COUNT(*) > 0 FROM documents));
SELECT setval('messages_id_seq', GREATEST((SELECT COALESCE(MAX(id), 0) FROM messages), 1), (SELECT COUNT(*) > 0 FROM messages));
SELECT setval('document_chunks_id_seq', GREATEST((SELECT COALESCE(MAX(id), 0) FROM document_chunks), 1), (SELECT COUNT(*) > 0 FROM document_chunks));

COMMIT;
