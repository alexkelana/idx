#!/bin/bash
# ============================================================
# SETUP DEPLOY STREAMLIT + GITHUB
# Double-click file ini di macOS untuk menjalankan.
# ============================================================

cd "$(dirname "$0")"

echo "============================================================"
echo "  IDX SCREENER — Setup Deploy Streamlit + GitHub"
echo "============================================================"
echo ""
echo "Folder kerja: $(pwd)"
echo ""

# --- 1. Cek Git ---
if ! command -v git &> /dev/null; then
    echo "❌ Git belum terinstall."
    echo "   Install dulu: xcode-select --install"
    echo "   atau download dari https://git-scm.com"
    read -p "Tekan Enter untuk keluar..."
    exit 1
fi
echo "✅ Git tersedia: $(git --version)"

# --- 2. Buat requirements.txt jika belum ada ---
if [ ! -f "requirements.txt" ]; then
    echo ""
    echo "📄 Membuat requirements.txt ..."
    cat > requirements.txt << 'EOF'
streamlit>=1.28.0
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
openpyxl>=3.1.0
lxml>=4.9.0
html5lib>=1.1
EOF
    echo "✅ requirements.txt dibuat."
else
    echo "✅ requirements.txt sudah ada."
fi

# --- 3. Buat .gitignore jika belum ada ---
if [ ! -f ".gitignore" ]; then
    echo ""
    echo "📄 Membuat .gitignore ..."
    cat > .gitignore << 'EOF'
*.csv
__pycache__/
*.pyc
.env
.venv/
venv/
*.xlsx
.DS_Store
.streamlit/secrets.toml
EOF
    echo "✅ .gitignore dibuat."
else
    echo "✅ .gitignore sudah ada."
fi

# --- 4. Cek file penting ---
echo ""
echo "📂 Cek file penting:"
REQUIRED_FILES=("dashboard.py" "master_screener_ai.py")
MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "   ✅ $f"
    else
        echo "   ❌ $f — TIDAK DITEMUKAN"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "⚠️  Ada file wajib yang hilang. Pastikan dashboard.py ada di folder ini."
    read -p "Tekan Enter untuk keluar..."
    exit 1
fi

# --- 5. Init Git (jika belum) ---
echo ""
if [ ! -d ".git" ]; then
    echo "🔧 Inisialisasi Git repository..."
    git init
    git branch -M main
    echo "✅ Git init selesai (branch: main)."
else
    echo "✅ Git repository sudah ada."
fi

# --- 6. Add & Commit ---
echo ""
echo "📦 Staging semua file..."
git add .

if git diff --cached --quiet; then
    echo "   (Tidak ada perubahan baru untuk di-commit)"
else
    git commit -m "Setup deploy Streamlit Cloud $(date +%Y-%m-%d)"
    echo "✅ Commit selesai."
fi

# --- 7. Remote GitHub ---
echo ""
echo "============================================================"
echo "  LANGKAH GITHUB"
echo "============================================================"
echo ""
echo "1. Buat repository BARU di: https://github.com/new"
echo "   - Repository name: idx-screener (atau nama lain)"
echo "   - Public"
echo "   - JANGAN centang 'Add README' (repo harus kosong)"
echo ""
echo "2. Setelah repo dibuat, GitHub akan menampilkan URL, contoh:"
echo "   https://github.com/USERNAME/idx-screener.git"
echo ""

# Cek apakah remote sudah ada
if git remote get-url origin &> /dev/null; then
    CURRENT_REMOTE=$(git remote get-url origin)
    echo "Remote 'origin' sudah ada: $CURRENT_REMOTE"
    read -p "Ganti remote? (y/N): " GANTI
    if [[ "$GANTI" =~ ^[Yy]$ ]]; then
        read -p "Masukkan URL repo GitHub: " REPO_URL
        git remote remove origin 2>/dev/null
        git remote add origin "$REPO_URL"
        echo "✅ Remote diganti."
    fi
else
    read -p "Masukkan URL repo GitHub: " REPO_URL
    if [ -n "$REPO_URL" ]; then
        git remote add origin "$REPO_URL"
        echo "✅ Remote 'origin' ditambahkan."
    else
        echo "⚠️  URL kosong. Skip push. Jalankan manual nanti:"
        echo "   git remote add origin https://github.com/USERNAME/idx-screener.git"
        echo "   git push -u origin main"
        read -p "Tekan Enter untuk keluar..."
        exit 0
    fi
fi

# --- 8. Push ---
echo ""
echo "🚀 Push ke GitHub..."
echo "   (Jika diminta login: pakai Personal Access Token, bukan password)"
echo ""

if git push -u origin main; then
    echo ""
    echo "============================================================"
    echo "  ✅ PUSH BERHASIL"
    echo "============================================================"
    echo ""
    echo "Langkah selanjutnya — Deploy di Streamlit Cloud:"
    echo ""
    echo "1. Buka: https://share.streamlit.io"
    echo "2. Login dengan GitHub"
    echo "3. Klik 'New app'"
    echo "4. Isi:"
    echo "   - Repository : (pilih repo yang baru di-push)"
    echo "   - Branch     : main"
    echo "   - Main file  : dashboard.py"
    echo "5. Klik Deploy"
    echo ""
    echo "Tunggu 1–3 menit, app akan live."
    echo "============================================================"
else
    echo ""
    echo "❌ Push gagal."
    echo ""
    echo "Kemungkinan penyebab:"
    echo "  - Belum login GitHub (pakai: gh auth login  atau  SSH key)"
    echo "  - URL repo salah"
    echo "  - Repo di GitHub belum dibuat / tidak kosong"
    echo ""
    echo "Coba manual:"
    echo "  git push -u origin main"
fi

echo ""
read -p "Tekan Enter untuk keluar..."