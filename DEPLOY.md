# Deploying this app — Streamlit Community Cloud (free)

Two steps. Neither needs the command line.

## 1. Put the code on GitHub

1. Go to **github.com/new**
2. Repository name: `banking-support-agent`
3. Set it to **Public** (Streamlit's free tier requires public repos)
4. Click **Create repository**
5. On the next screen click **"uploading an existing file"**
6. Drag in everything from this folder EXCEPT:
   - `.env`  — never upload this, it has your key
   - `support.db`  — regenerates itself
7. Click **Commit changes**

`.streamlit` is a hidden folder. If Windows doesn't show it, turn on
View → Hidden items, or skip it — the app runs fine without it.

## 2. Deploy

1. Go to **share.streamlit.io** and sign in with the same GitHub account
2. Click **Create app** → deploy from GitHub
3. Repository: `strongj3369/banking-support-agent`
4. Branch: `main`
5. Main file path: `app.py`
6. Click **Advanced settings** BEFORE deploying
7. In the **Secrets** box paste exactly this, with your real key:

   ANTHROPIC_API_KEY = "sk-ant-..."

   Your key is in signal-app\scraper\.env. Streamlit stores secrets
   encrypted — they are not visible in the public repo.
8. Click **Deploy**

First build takes 2-3 minutes. You get a URL like
https://banking-support-agent.streamlit.app

## Cost protection already built in

- 15 live routing calls per browser session
- 200 model calls per day total, across all visitors
- 500-character cap per message

When a limit is hit, the Tickets, Logs and Evaluation tabs keep working — only
live routing pauses. Haiku is a fraction of a cent per call, so the daily
ceiling is a few cents.

## Taking it down later

share.streamlit.io -> your app -> the three-dot menu -> Delete app.
