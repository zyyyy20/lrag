-- Add persisted tool results for ReAct-generated chat artifacts.
ALTER TABLE messages
    ADD COLUMN IF NOT EXISTS tool_results JSONB;

CREATE INDEX IF NOT EXISTS ix_messages_tool_results_gin
    ON messages USING GIN (tool_results)
    WHERE tool_results IS NOT NULL;
