const revealListItem = new WeakMap();
function initLists(root = document) {
root.querySelectorAll('[data-list]').forEach(list => {
  if (list.dataset.initialized) return;
  list.dataset.initialized = 'true';
  const items = [...list.querySelectorAll('[data-item]')];
  const search = list.querySelector('input[type="search"]');
  const selects = [...list.querySelectorAll('select')];
  const showSmoke = list.querySelector('[data-show-smoke]');
  const showComparison = list.querySelector('[data-show-comparison]');
  const size = Number(list.dataset.pageSize || 20);
  const previous = list.querySelector('[data-previous]');
  const next = list.querySelector('[data-next]');
  const texts = items.map(item => item.textContent.toLocaleLowerCase());
  let page = Number(list.dataset.currentPage || 0);
  function matchSelect(item, select) {
    const value = select.value || 'all';
    if (value === 'all') return true;
    if (value === 'promoted') return item.dataset.promoted === 'true';
    const attr = select.dataset.filterAttr || 'kind';
    return item.dataset[attr] === value;
  }
  function visible(item) {
    if (showSmoke && !showSmoke.checked && item.dataset.kind === 'smoke') return false;
    if (showComparison && !showComparison.checked && item.dataset.kind === 'comparison') return false;
    return true;
  }
  function render(reset = false) {
    if (reset) page = 0;
    const query = (search?.value || '').trim().toLocaleLowerCase();
    const matches = items.filter((item, index) => (!query || texts[index].includes(query))
        && visible(item) && selects.every(select => matchSelect(item, select)));
    const pages = Math.max(1, Math.ceil(matches.length / size));
    page = Math.min(page, pages - 1);
    list.dataset.currentPage = page;
    const visibleItems = new Set(matches.slice(page * size, (page + 1) * size));
    items.forEach(item => { item.hidden = !visibleItems.has(item); });
    list.querySelector('[data-list-count]').textContent = `共 ${matches.length} 条${matches.length !== items.length ? `，从 ${items.length} 条中筛选` : ''}`;
    list.querySelector('[data-page-label]').textContent = `${page + 1} / ${pages}`;
    list.querySelector('[data-no-results]').hidden = matches.length > 0;
    previous.disabled = page === 0;
    next.disabled = page >= pages - 1;
  }
  search?.addEventListener('input', () => render(true));
  selects.forEach(select => select.addEventListener('change', () => render(true)));
  showSmoke?.addEventListener('change', () => render(true));
  showComparison?.addEventListener('change', () => render(true));
  previous.addEventListener('click', () => { page--; render(); });
  next.addEventListener('click', () => { page++; render(); list.scrollIntoView({block:'start', behavior:'smooth'}); });
  revealListItem.set(list, item => {
    const index = items.indexOf(item);
    if (index < 0) return;
    if (search) search.value = '';
    selects.forEach(select => { select.value = 'all'; });
    page = Math.floor(index / size);
    render();
  });
  render();
});
}

let revealedCaseHash = '';
function revealLinkedCase(force = false) {
  const hash = location.hash;
  if (!hash.startsWith('#case-') || (!force && hash === revealedCaseHash)) return;
  // ID 与链接都由同一个 Case ID 编码生成，不将片段拼进 CSS 选择器。
  const item = document.getElementById(hash.slice(1));
  if (!item?.matches('.case')) return;
  revealListItem.get(item.closest('[data-list]'))?.(item);
  item.open = true;
  item.querySelector(':scope > summary').focus({preventScroll:true});
  item.scrollIntoView({block:'start'});
  revealedCaseHash = hash;
}
async function copyCaseLink(button) {
  const link = button.closest('.case').querySelector('[data-case-link]');
  try {
    await navigator.clipboard.writeText(link.href);
    button.textContent = '已复制';
  } catch (error) {
    // 普通 HTTP 或剪贴板权限被拒时，仍可手动复制完整链接。
    window.prompt('复制本题链接', link.href);
  }
  setTimeout(() => { button.textContent = '复制链接'; }, 2000);
}

const lastHtml = new WeakMap();
function detailKey(element) {
  const caseId = element.closest('[data-case-id]')?.dataset.caseId || '';
  const traceId = element.closest('[data-trace-url]')?.dataset.traceUrl || '';
  const scope = [];
  for (let parent = element.parentElement; parent; parent = parent.parentElement) {
    if (parent.dataset.detailKey) scope.push(parent.dataset.detailKey);
  }
  return `${caseId}:${traceId}:${scope.join('/')}:${element.dataset.detailKey || (element.matches('.case') ? 'case' : element.querySelector('summary')?.textContent.trim())}`;
}
function updateRegion(region, html) {
  if (lastHtml.get(region) === html) return;
  const open = new Set([...region.querySelectorAll('details[open]')].map(detailKey));
  const fields = [...region.querySelectorAll('input,select')].map(el => ({value:el.value, checked:el.type === 'checkbox' ? el.checked : undefined, focused:el === document.activeElement, start:el.selectionStart, end:el.selectionEnd}));
  const pages = [...region.querySelectorAll('[data-list]')].map(el => el.dataset.currentPage);
  const scrolls = [...region.querySelectorAll('.chat,pre')].map(el => el.scrollTop);
  const y = window.scrollY;
  // 逐题列表更新时复用已加载的实录节点，保留内部展开状态及滚动位置。
  const traces = new Map([...region.querySelectorAll('[data-trace-url]')].map(el => [el.dataset.traceUrl, el]));
  const parsed = new DOMParser().parseFromString(html, 'text/html');
  region.replaceChildren(...parsed.body.childNodes);
  region.querySelectorAll('[data-trace-url]').forEach(el => {
    if (traces.has(el.dataset.traceUrl)) el.replaceWith(traces.get(el.dataset.traceUrl));
  });
  region.querySelectorAll('details').forEach(el => { el.open = open.has(detailKey(el)); });
  region.querySelectorAll('input,select').forEach((el, i) => {
    if (!fields[i]) return;
    el.value = fields[i].value;
    if (fields[i].checked !== undefined) el.checked = fields[i].checked;
    if (fields[i].focused) {
      el.focus({preventScroll:true});
      if (el.setSelectionRange && fields[i].start !== null) el.setSelectionRange(fields[i].start, fields[i].end);
    }
  });
  region.querySelectorAll('[data-list]').forEach((el, i) => { el.dataset.currentPage = pages[i] || 0; });
  initLists(region);
  region.querySelectorAll('.chat,pre').forEach((el, i) => { el.scrollTop = scrolls[i] || 0; });
  window.scrollTo(window.scrollX, y);
  lastHtml.set(region, html);
}
function traceVisible(node) {
  if (!node.isConnected || !node.open) return false;
  for (let parent = node.parentElement; parent; parent = parent.parentElement) {
    if (parent.hidden || (parent.matches('details') && !parent.open)) return false;
  }
  return true;
}
async function refreshTrace(node) {
  if (!traceVisible(node) || node.dataset.loading || node.dataset.finished) return;
  node.dataset.loading = 'true';
  const status = node.querySelector('[data-trace-status]');
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const url = new URL(node.dataset.traceUrl, location.href);
    url.searchParams.set('revision', node.dataset.revision || '');
    const response = await fetch(url, {cache:'no-store', signal:controller.signal});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    if (!node.isConnected) return;
    if (payload.html !== undefined) updateRegion(node.querySelector('[data-trace-body]'), payload.html);
    node.dataset.revision = payload.revision;
    if (payload.status !== 'running') node.dataset.finished = 'true';
    status.textContent = payload.status === 'running' ? '每 3 秒同步本题调用记录；等待中的步骤保留已发送输入。' : '本次调用记录已加载，可逐层展开查看。';
  } catch (error) {
    status.textContent = '调用记录暂时无法加载，保留已显示内容；正在自动重试。';
  } finally {
    clearTimeout(timeout);
    delete node.dataset.loading;
  }
}
function refreshOpenTraces() {
  document.querySelectorAll('[data-trace-url]').forEach(refreshTrace);
}
function updateClocks() {
  document.querySelectorAll('[data-stage-clock]').forEach(el => {
    const seconds = Math.max(0, Math.floor(Date.now() / 1000 - Number(el.dataset.stageClock)));
    el.textContent = `本步骤已用时 ${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`;
  });
}
let refreshing = false;
let casesRevision = '';
let configRevision = '';
async function refreshLive() {
  const endpoint = document.body.dataset.liveUrl;
  if (!endpoint || refreshing) return;
  refreshing = true;
  const status = document.querySelector('[data-live-status]');
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const query = new URLSearchParams({cases_revision:casesRevision, config_revision:configRevision});
    const response = await fetch(`${endpoint}?${query}`, {cache:'no-store', signal:controller.signal});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    document.querySelectorAll('[data-live-region]').forEach(region => {
      const html = payload.regions[region.dataset.liveRegion];
      if (html !== undefined) updateRegion(region, html);
    });
    casesRevision = payload.cases_revision || casesRevision;
    configRevision = payload.config_revision || configRevision;
    // 链接指向的题可能稍后才产生记录；首次出现时定位，后续轮询保留用户的浏览位置。
    revealLinkedCase();
    updateClocks();
    refreshOpenTraces();
    status.textContent = `实时更新 · 每 3 秒同步 · 最近同步 ${new Date().toLocaleTimeString('zh-CN', {hour12:false})}`;
    status.classList.remove('danger-text');
  } catch (error) {
    status.textContent = '更新暂时中断，保留上次数据；正在自动重试';
    status.classList.add('danger-text');
  } finally {
    clearTimeout(timeout);
    refreshing = false;
  }
}
initLists();
revealLinkedCase();
window.addEventListener('hashchange', () => { revealedCaseHash = ''; revealLinkedCase(); });
document.addEventListener('toggle', event => {
  if (event.target.matches('[data-trace-url],.case')) refreshOpenTraces();
}, true);
document.addEventListener('click', event => {
  const copy = event.target.closest('[data-copy-case-link]');
  if (copy) {
    event.preventDefault();
    copyCaseLink(copy);
    return;
  }
  const link = event.target.closest('[data-case-link]');
  if (link && event.button === 0 && !event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey) {
    event.preventDefault();
    if (location.hash !== link.hash) history.pushState(null, '', link.href);
    revealLinkedCase(true);
    return;
  }
  if (event.target.closest('[data-reload]')) {
    if (document.body.dataset.liveUrl) refreshLive();
    else location.reload();
  }
});
if (document.body.dataset.liveUrl) {
  refreshLive();
  setInterval(() => { if (!document.hidden) refreshLive(); }, 3000);
  setInterval(updateClocks, 1000);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refreshLive(); });
}
