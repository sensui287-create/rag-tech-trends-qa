/* Shared browser/Node SSE framing. No UI, network or model dependencies. */
(function (root) {
    'use strict';

    function parseSseBlock(block) {
        let event = 'message';
        const data = [];
        for (const line of block.split(/\r?\n/)) {
            if (line.startsWith('event:')) event = line.slice(6).replace(/^ /, '');
            else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
        }
        return data.length ? { event, data: data.join('\n') } : null;
    }

    function createSseDecoder() {
        let buffer = '';
        return {
            push(text, final = false) {
                buffer += text;
                const events = [];
                let separator;
                while ((separator = /\r?\n\r?\n/.exec(buffer)) !== null) {
                    const parsed = parseSseBlock(buffer.slice(0, separator.index));
                    buffer = buffer.slice(separator.index + separator[0].length);
                    if (parsed) events.push(parsed);
                }
                if (final && buffer) {
                    const parsed = parseSseBlock(buffer);
                    if (parsed) events.push(parsed);
                    buffer = '';
                }
                return events;
            }
        };
    }

    const api = { parseSseBlock, createSseDecoder };
    if (typeof module === 'object' && module.exports) module.exports = api;
    else root.LongChainStream = api;
}(typeof globalThis === 'object' ? globalThis : this));
