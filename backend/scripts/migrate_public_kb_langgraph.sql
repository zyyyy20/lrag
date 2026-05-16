-- Public knowledge base + LangGraph migration.
-- LangGraph checkpoint tables are created by PostgresSaver.setup() at runtime.

ALTER TABLE knowledge_bases
    ADD COLUMN IF NOT EXISTS created_by INTEGER NULL;

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS created_by INTEGER NULL;

UPDATE knowledge_bases
SET created_by = user_id
WHERE created_by IS NULL;

UPDATE documents
SET created_by = user_id
WHERE created_by IS NULL;

CREATE INDEX IF NOT EXISTS ix_knowledge_bases_created_by
    ON knowledge_bases (created_by);

CREATE INDEX IF NOT EXISTS ix_documents_created_by
    ON documents (created_by);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_knowledge_bases_created_by_users'
    ) THEN
        ALTER TABLE knowledge_bases
            ADD CONSTRAINT fk_knowledge_bases_created_by_users
            FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_documents_created_by_users'
    ) THEN
        ALTER TABLE documents
            ADD CONSTRAINT fk_documents_created_by_users
            FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL;
    END IF;
END $$;
