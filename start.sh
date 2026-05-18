#!/bin/bash
# Запуск тренажёра госэкзаменов

echo "╔══════════════════════════════════════════╗"
echo "║        ЭкзаменАИ — Тренажёр              ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check for API key in env
if [ -z "$MISTRAL_API_KEY" ]; then
  echo "⚠  MISTRAL_API_KEY не задан в окружении."
  echo "   Можно задать ключ прямо в интерфейсе приложения."
  echo ""
fi

# Install deps if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "📦 Устанавливаю зависимости..."
  pip3 install -r requirements.txt -q
fi

echo "🚀 Запускаю сервер на http://localhost:8000"
echo "   Нажмите Ctrl+C для остановки"
echo ""

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
