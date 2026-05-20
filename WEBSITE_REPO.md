# Website moved to its own repository

The Touchless marketing website (**touchless-control.com**) is no longer
part of this monorepo. It lives in a dedicated repo with Cloudflare Pages
git auto-deploy.

- **GitHub:** https://github.com/markovk-tini/touchless-website
- **Local clone:** `c:\touchless-website`
- **Deploy:** push to `main` → Cloudflare Pages auto-builds → live on
  touchless-control.com in ~60s. No manual upload, no cache purge (a
  `_headers` file forces CSS/JS/HTML revalidation).

The previous `hgr-download-page/` folder was removed from this repo to
keep a single source of truth. Make all website edits in the dedicated
repo above, not here.

<!-- Author: Konstantin Markov -->
