import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

// ──────────────────────────────────────────────
// 타입
// ──────────────────────────────────────────────
interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

interface Citation {
  chunk_id: string
  header: string
  source_file: string
}

interface RagResult {
  answer: string
  citations: Citation[]
  fallback: boolean
  fallback_reason: string | null
  reformulated_query: string | null
}

interface Conversation {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: number
}

// ──────────────────────────────────────────────
// localStorage 헬퍼
// ──────────────────────────────────────────────
const STORAGE_KEY = 'autodraft_conversations'

function loadConversations(): Conversation[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
  } catch {
    return []
  }
}

function saveConversations(convs: Conversation[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(convs))
}

// ──────────────────────────────────────────────
// API 호출
// ──────────────────────────────────────────────
async function callChat(query: string, history: ChatMessage[]): Promise<RagResult> {
  const res = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      chat_history: history.length > 0 ? history : null,
    }),
  })
  if (!res.ok) throw new Error(`서버 오류 ${res.status}`)
  return res.json()
}

// ──────────────────────────────────────────────
// 컴포넌트
// ──────────────────────────────────────────────
export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations)
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [lastResult, setLastResult] = useState<RagResult | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [searchMode, setSearchMode] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editingTitle, setEditingTitle] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  function startEditTitle(conv: Conversation, e: React.MouseEvent) {
    e.stopPropagation()
    setEditingId(conv.id)
    setEditingTitle(conv.title)
  }

  function commitEditTitle(id: string) {
    const trimmed = editingTitle.trim()
    if (trimmed) {
      const updated = conversations.map(c => c.id === id ? { ...c, title: trimmed } : c)
      setConversations(updated)
      saveConversations(updated)
    }
    setEditingId(null)
  }

  function handleEditKey(e: React.KeyboardEvent, id: string) {
    if (e.key === 'Enter') { e.preventDefault(); commitEditTitle(id) }
    if (e.key === 'Escape') setEditingId(null)
  }

  function formatDate(ts: number): string {
    const now = new Date()
    const d = new Date(ts)
    const diffDays = Math.floor((now.setHours(0,0,0,0) - d.setHours(0,0,0,0)) / 86400000)
    if (diffDays === 0) return '오늘'
    if (diffDays === 1) return '어제'
    const orig = new Date(ts)
    return `${orig.getMonth() + 1}/${orig.getDate()}`
  }

  const visibleConvs = searchMode && searchQuery.trim()
    ? conversations.filter(c => c.title.includes(searchQuery.trim()))
    : conversations

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    if (!loading) inputRef.current?.focus()
  }, [loading])

  function startNewChat() {
    setActiveId(null)
    setMessages([])
    setInput('')
    setLastResult(null)
  }

  function openConversation(conv: Conversation) {
    setActiveId(conv.id)
    setMessages(conv.messages)
    setLastResult(null)
  }

  function deleteConversation(id: string, e: React.MouseEvent) {
    e.stopPropagation()
    const updated = conversations.filter(c => c.id !== id)
    setConversations(updated)
    saveConversations(updated)
    if (activeId === id) startNewChat()
  }

  async function send() {
    const query = input.trim()
    if (!query || loading) return

    const userMsg: ChatMessage = { role: 'user', content: query }
    const nextHistory = [...messages, userMsg]

    setMessages(nextHistory)
    setInput('')
    setLoading(true)
    setLastResult(null)

    try {
      const result = await callChat(query, messages)
      setLastResult(result)
      const finalMessages: ChatMessage[] = [
        ...nextHistory,
        { role: 'assistant', content: result.answer },
      ]
      setMessages(finalMessages)

      // 대화 저장: 첫 메시지면 신규 생성, 이미 있으면 업데이트
      const convId = activeId ?? crypto.randomUUID()
      const title = query.length > 28 ? query.slice(0, 28) + '…' : query

      setConversations(prev => {
        const exists = prev.some(c => c.id === convId)
        const updated = exists
          ? prev.map(c => c.id === convId ? { ...c, messages: finalMessages } : c)
          : [{ id: convId, title, messages: finalMessages, createdAt: Date.now() }, ...prev]
        saveConversations(updated)
        return updated
      })

      if (!activeId) setActiveId(convId)
    } catch {
      setMessages([
        ...nextHistory,
        { role: 'assistant', content: '오류가 발생했습니다. 서버 상태를 확인해주세요.' },
      ])
    } finally {
      setLoading(false)
    }
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="app-shell">
      {/* ── 사이드바 ── */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-top-bar">
          <button className="new-chat-btn" onClick={startNewChat}>
            + 새 채팅
          </button>
          <button
            className={`search-btn ${searchMode ? 'active' : ''}`}
            onClick={() => { setSearchMode(v => !v); setSearchQuery('') }}
            title="검색"
          >
            🔍
          </button>
        </div>

        {searchMode && (
          <input
            className="search-input"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="대화 검색..."
            autoFocus
          />
        )}

        <hr className="conv-divider" />
        <div className="conv-list">
          {visibleConvs.length === 0 && (
            <p className="conv-empty">
              {searchMode && searchQuery ? '검색 결과 없음' : '대화 기록이 없습니다.'}
            </p>
          )}
          {visibleConvs.map(conv => (
            <div
              key={conv.id}
              className={`conv-item ${activeId === conv.id ? 'active' : ''}`}
              onClick={() => editingId !== conv.id && openConversation(conv)}
            >
              {editingId === conv.id ? (
                <input
                  className="conv-title-input"
                  value={editingTitle}
                  onChange={e => setEditingTitle(e.target.value)}
                  onBlur={() => commitEditTitle(conv.id)}
                  onKeyDown={e => handleEditKey(e, conv.id)}
                  onClick={e => e.stopPropagation()}
                  autoFocus
                />
              ) : (
                <span
                  className="conv-title"
                  onDoubleClick={e => startEditTitle(conv, e)}
                  title="더블클릭하여 제목 수정"
                >
                  {conv.title}
                </span>
              )}
              {searchMode && editingId !== conv.id && (
                <span className="conv-date">{formatDate(conv.createdAt)}</span>
              )}
              <button
                className="conv-delete"
                onClick={e => deleteConversation(conv.id, e)}
                title="삭제"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* ── 메인 채팅 ── */}
      <div className="main">
        <header className="header">
          <button className="sidebar-toggle" onClick={() => setSidebarOpen(v => !v)} title="메뉴">
            <span /><span /><span />
          </button>
          <span className="logo">AutoDraft</span>
          {activeId && messages.length > 0 && (
            <span className="chat-title">
              {conversations.find(c => c.id === activeId)?.title}
            </span>
          )}
        </header>

        {messages.length === 0 ? (
          /* ── 빈 상태: 입력창 중앙 ── */
          <div className="chat-empty-state">
            <p className="empty-hint">사내 문서에 대해 질문해보세요.</p>
            <div className="query-guide">
              <p className="guide-title">💡 이런 질문을 해보세요</p>
              <ul className="guide-list">
                <li>등록된 파일이 궁금하면 &nbsp;<span className="guide-example">"파일 몇 개 있어?"</span></li>
                <li>특정 주제 자료가 있는지 &nbsp;<span className="guide-example">"~ 관련 자료 있어?"</span></li>
                <li>문서 내용이 궁금하면 &nbsp;<span className="guide-example">"~의 핵심 기술이 뭐야?"</span></li>
                <li>이전 답변 이어서 &nbsp;<span className="guide-example">"그럼 거기서 ~은?"</span></li>
              </ul>
            </div>
            <div className="input-row">
              <textarea
                ref={inputRef}
                className="input"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="질문을 입력하세요  (Enter 전송 / Shift+Enter 줄바꿈)"
                rows={1}
                disabled={loading}
              />
              <button className="send-btn" onClick={send} disabled={loading || !input.trim()}>
                전송
              </button>
            </div>
          </div>
        ) : (
          /* ── 채팅 중: 메시지 + 하단 입력창 ── */
          <>
            <main className="chat-area">
              {messages.map((msg, i) => (
                <div key={i} className={`bubble-row ${msg.role}`}>
                  <div className="bubble">
                    {msg.role === 'assistant'
                      ? <ReactMarkdown>{msg.content}</ReactMarkdown>
                      : msg.content}
                  </div>
                </div>
              ))}

              {lastResult && !lastResult.fallback && lastResult.citations.length > 0 && (
                <div className="citations">
                  {lastResult.citations.map((c, i) => (
                    <span key={i} className="citation-tag" title={c.source_file}>
                      {c.header || c.source_file}
                    </span>
                  ))}
                </div>
              )}

              {loading && (
                <div className="bubble-row assistant">
                  <div className="bubble loading">
                    <span /><span /><span />
                  </div>
                </div>
              )}

              <div ref={bottomRef} />
            </main>

            <footer className="input-bar">
              <div className="input-row">
                <textarea
                  ref={inputRef}
                  className="input"
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKey}
                  placeholder="질문을 입력하세요  (Enter 전송 / Shift+Enter 줄바꿈)"
                  rows={1}
                  disabled={loading}
                />
                <button className="send-btn" onClick={send} disabled={loading || !input.trim()}>
                  전송
                </button>
              </div>
            </footer>
          </>
        )}
      </div>
    </div>
  )
}
