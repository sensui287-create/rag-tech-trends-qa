"""Test the browser's real SSE decoder through Node, without CDN or browser IO."""

from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for frontend framing checks")
    def test_stream_framing_for_all_boundaries_comments_multiline_and_eof(self):
        script = r'''
const assert = require('node:assert/strict');
const { createSseDecoder, parseSseBlock } = require('./backend/static/stream.js');
const wire = ':ping\r\n\r\nevent: answer_chunk\r\ndata: "中文"\r\n\r\nevent: done\r\ndata: [DONE]\r\n\r\n';
const expected = [{event: 'answer_chunk', data: '"中文"'}, {event: 'done', data: '[DONE]'}];
for (let split = 0; split <= wire.length; split++) {
    const decoder = createSseDecoder();
    assert.deepEqual([...decoder.push(wire.slice(0, split)), ...decoder.push(wire.slice(split), true)], expected);
}
const decoder = createSseDecoder();
const singleCharacters = [...wire].flatMap(char => decoder.push(char));
assert.deepEqual(singleCharacters, expected);
assert.deepEqual(parseSseBlock('event: detail\ndata: first\ndata: second'), {event: 'detail', data: 'first\nsecond'});
assert.deepEqual(createSseDecoder().push('event: answer\ndata: "last"', true), [{event: 'answer', data: '"last"'}]);
assert.equal(parseSseBlock(':heartbeat'), null);
'''
        subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT, check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
