# AutoDraft 챗봇 UI

> AutoDraft RAG 시스템의 프론트엔드. React + TypeScript + Vite 기반 챗봇 인터페이스.

---

## 기술 스택

| 항목 | 버전 |
|------|------|
| React | 19 |
| TypeScript | 5 |
| Vite | 6 |
| react-markdown | 최신 |

---

## 주요 기능

- 사용자 질문 입력 → `POST /chat` API 호출 → 답변 표시
- 마크다운 렌더링 (볼드, 목록 등)
- 출처(citation) 태그 표시
- 로딩 애니메이션
- 대화 기록 localStorage 저장/복원/삭제
- 사이드바 대화 목록 (토글, 스크롤, 검색, 날짜 표시)
- 채팅방 제목 더블클릭 인라인 편집
- 빈 상태: 입력창 화면 중앙 배치 → 첫 질문 후 하단 이동
- 응답 완료 후 입력창 자동 포커스

---

## 파일 구조

```
frontend/
├── src/
│   ├── App.tsx        # 챗봇 UI 메인 컴포넌트
│   ├── App.css        # 챗봇 UI 스타일
│   ├── index.css      # 전역 CSS 변수 및 기반 스타일
│   └── main.tsx       # React 앱 진입점
├── public/
├── package.json
└── vite.config.ts
```

---

## 실행 방법

```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
# → http://localhost:5173
```

> 백엔드 API 서버(`src/api.py`)가 `http://localhost:8000`에서 먼저 실행 중이어야 합니다.

---

## API 연동

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/chat` | POST | 질문 전송 → RAG 답변 수신 |
| `/health` | GET | 서버 상태 확인 |

백엔드 실행 방법은 상위 디렉토리의 [README.md](../README.md)를 참고하세요.
