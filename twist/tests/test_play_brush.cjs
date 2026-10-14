// Unit checks for the 64px brush and playback, without launching a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

async function main() {
  const page = fs.readFileSync(path.join(__dirname, '..', 'play.html'), 'utf8');
  const source = page.match(/<script>([\s\S]*?)<\/script>/)[1];
  const elements = new Map();
  for (const match of page.matchAll(/id="([^"]+)"/g)) {
    const listeners = {}, classes = new Set();
    const element = {value: '', textContent: '', disabled: false, hidden: false, listeners,
      addEventListener(name, callback) { listeners[name] = callback; },
      setAttribute(name, value) { this[name] = value; }, focus() {}, setPointerCapture() {},
      removeAttribute(name) { delete this[name]; },
      classList: {add(name) {classes.add(name);}, remove(name) {classes.delete(name);},
        toggle(name, value) {if(value) classes.add(name); else classes.delete(name);}, contains(name) {return classes.has(name);}},
      getBoundingClientRect() {return {left:0,top:0,width:320,height:320};},
      getContext() {return {createImageData(width,height) {return {data:new Uint8ClampedArray(width*height*4)};},
        putImageData(image) {element.pixels = image.data;}};}};
    elements.set(match[1], element);
  }
  const el = name => elements.get(name);
  Object.assign(el('width'), {value:'4'}); Object.assign(el('softness'), {value:'.6'});
  Object.assign(el('frames'), {value:'21'}); Object.assign(el('fps'), {value:'12'});
  const requests = [], timers = new Map(); let timerIndex = 0;
  const fakeAnimation = {frames:['start','middle','end'],times:[0,.5,1],sequence:[0,1,2,1],
    durations:[600,80,600,80],gif:'data:image/gif;base64,R0lG',seconds:.2};
  const context = vm.createContext({document:{getElementById:el,addEventListener() {}},
    setTimeout(callback) {timers.set(++timerIndex,callback);return timerIndex;},
    clearTimeout(index) {timers.delete(index);},
    fetch: async (url, options) => {
      if (url === '/api/info') return {ok:true,json:async()=>({checkpoint:'best.pt',step:123})};
      requests.push(JSON.parse(options.body)); return {ok:true,json:async()=>fakeAnimation};
    }});
  vm.runInContext(source, context);
  const run = code => vm.runInContext(code, context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(el('generate').disabled, false);

  // A soft stroke has dark ink, untouched background, and intermediate gray.
  run("left.stroke({x:14,y:31.7},{x:50,y:31.7},4,.6,false);left.paint()");
  const pixels = Array.from(run('left.values()'));
  assert(pixels.includes(0) && pixels.includes(255) && pixels.some(value=>value>0 && value<255));
  const previous = Array.from(run('left.ink'));
  run("left.stroke({x:14,y:31.7},{x:50,y:31.7},4,.6,false)");
  assert.deepEqual(Array.from(run('left.ink')), previous, 'repeated samples must preserve the soft edge');
  run("left.remember();left.stroke({x:14,y:31.7},{x:50,y:31.7},4,.6,true);left.undo()");
  assert.deepEqual(Array.from(run('left.ink')), previous, 'undo must restore erased grayscale coverage');

  // Pointer coordinates are transformed into the 64px image; one drag fills a line.
  const event = (x,y) => ({pointerId:1,pointerType:'mouse',button:0,clientX:x,clientY:y,preventDefault(){}});
  el('right').listeners.pointerdown(event(100,60));
  el('right').listeners.pointermove(event(100,260));
  el('right').listeners.pointerup(event(100,260));
  assert(run('right.ink[30*64+20]') > .9);
  el('undo-right').listeners.click();
  assert.equal(run('right.ink.some(value=>value>0)'), false);

  const before = Array.from(run('left.ink'));
  el('swap').listeners.click();
  assert.deepEqual(Array.from(run('right.ink')), before);
  el('example').listeners.click();
  await el('generate').listeners.click();
  assert.equal(requests.length, 1);
  assert.equal(requests[0].source.length, 4096);
  assert.equal(requests[0].target.length, 4096);
  assert.equal(requests[0].frames, 21);
  assert.equal(el('result').src, 'start');
  assert.equal(el('download').href, fakeAnimation.gif);
  assert.equal(el('generate').disabled, false);
  assert.equal(el('play').disabled, false);
  el('timeline').value = '1';el('timeline').listeners.input();
  assert.equal(el('result').src, 'middle');assert.equal(el('time').textContent, 't = 0.50');
  assert.equal(el('play').textContent, '播放');
  el('play').listeners.click();
  assert.equal(el('play').textContent, '暂停');
  const assertClearedTransition = () => {
    assert.equal(run('animation'), null);
    assert.equal(run('playing'), false);
    assert.equal(timers.size, 0, 'clearing must cancel animation timers');
    assert.equal(el('result').hidden, true);
    assert.equal(el('result').src, undefined);
    assert.equal(el('empty').hidden, false);
    assert.equal(el('timeline').disabled, true);
    assert.equal(Number(el('timeline').value), 0);
    assert.equal(el('time').textContent, 't = 0.00');
    assert.equal(el('play').disabled, true);
    assert.equal(el('download').href, undefined);
    assert.equal(el('download')['aria-disabled'], 'true');
    assert(el('download').classList.contains('disabled'));
  };
  el('clear-left').listeners.click();
  assertClearedTransition();
  const count = requests.length;
  await el('generate').listeners.click();
  assert.equal(requests.length, count, 'blank drawing must not submit inference');
  assert(el('status').classList.contains('error'));
  for (const button of ['clear-right', 'clear-both']) {
    el('example').listeners.click();
    await el('generate').listeners.click();
    assert.equal(el('result').hidden, false);
    el(button).listeners.click();
    assertClearedTransition();
  }
  console.log('PASS: soft brush, overlap, eraser undo, pointer drag, swap, example, request, playback, scrub, blank validation, clearing all transitions');
  if (process.env.TWIST_PLAY_PAYLOAD) fs.writeFileSync(process.env.TWIST_PLAY_PAYLOAD, JSON.stringify(requests[0]));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
