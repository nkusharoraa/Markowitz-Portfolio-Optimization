# Deploying to GitHub Pages

## Quick Setup (5 minutes)

### Step 1: Push to GitHub

```bash
# Initialize git repository (if not already done)
git init
git add .
git commit -m "Initial commit with GitHub Pages support"

# Create GitHub repository and push
git remote add origin https://github.com/nkusharoraa/Markowitz-Portfolio-Optimization.git
git branch -M main
git push -u origin main
```

### Step 2: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings**
3. Scroll to **Pages** section (in left sidebar)
4. Under **Source**, select:
   - Branch: `main`
   - Folder: `/docs`
5. Click **Save**

### Step 3: Wait for Deployment

- GitHub will automatically build and deploy
- Usually takes 1-2 minutes
- Check the green checkmark in Actions tab

### Step 4: Access Your Site

Your site will be live at:
```
https://nkusharoraa.github.io/Markowitz-Portfolio-Optimization/
```

---

## Features of the Web App

✨ **Interactive Portfolio Optimization**
- Run optimizations directly in your browser
- No installation required
- Works offline after first load

📊 **Two Input Methods**
1. **Manual Entry:** Enter returns and covariance manually
2. **CSV Upload:** Upload historical prices (auto-calculates parameters)

🎨 **Modern UI**
- Responsive design (mobile-friendly)
- Gradient backgrounds
- Professional color scheme
- Smooth animations

📈 **Real-Time Calculations**
- Instant results
- No server needed
- Pure JavaScript implementation

---

## File Structure

```
docs/
├── index.html              # Main web application
├── README.md               # This file
├── cli-guide.md           # CLI documentation
└── covariance_estimation.md  # Technical docs
```

---

## Customization

### Update Repository URLs

Edit `docs/index.html` and update the footer links:

```html
<a href="https://github.com/nkusharoraa/Markowitz-Portfolio-Optimization">GitHub</a>
```

### Change Color Scheme

Edit the CSS variables in `docs/index.html`:

```css
:root {
    --primary: #2E86AB;      /* Change to your color */
    --secondary: #A23B72;    /* Change to your color */
    --success: #06A77D;
    /* ... */
}
```

### Add Google Analytics (Optional)

Add before `</head>`:

```html
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

---

## Testing Locally

### Using Python's Built-in Server

```bash
cd docs
python -m http.server 8000
```

Then open: `http://localhost:8000`

### Using VS Code Live Server

1. Install "Live Server" extension
2. Right-click `docs/index.html`
3. Select "Open with Live Server"

---

## Limitations & Notes

### Current Implementation

The web version uses a **simplified optimization algorithm** for demonstration purposes. For production use cases:

✅ **Use for:**
- Quick demonstrations
- Educational purposes
- Preliminary analysis
- Portfolio visualization

⚠️ **For serious analysis, use the Python CLI:**
```bash
python optimize.py your_data.csv
```

The Python version uses proper convex optimization (CVXPY) with guaranteed optimal solutions.

### Future Enhancements

Potential improvements for the web version:
- [ ] Integration with proper optimization library (e.g., GLPK.js)
- [ ] Chart.js for interactive visualizations
- [ ] Historical data API integration
- [ ] Save/load portfolio configurations
- [ ] Export results as PDF

---

## Troubleshooting

### Pages not updating

1. Clear browser cache
2. Wait a few minutes (GitHub Pages cache)
3. Check Actions tab for build errors
4. Ensure `/docs` folder is committed

### 404 Error

- Verify Source is set to `/docs` folder
- Check that `index.html` exists in `docs/`
- Repository must be public (or GitHub Pro for private)

### Calculations seem off

Remember: The web version uses simplified calculations. For precise results, use the Python CLI with proper optimization libraries.

---

## Support

- **Python CLI:** Full documentation in main README
- **Web Issues:** Check browser console for errors
- **GitHub Pages:** [Official Documentation](https://pages.github.com/)

---

## Example Custom Domain (Optional)

### Add Custom Domain

1. Create file `docs/CNAME` containing:
   ```
   portfolio.yourdomain.com
   ```

2. Add DNS record at your domain registrar:
   ```
   Type: CNAME
   Host: portfolio
   Value: nkusharoraa.github.io
   ```

3. Enable HTTPS in GitHub Pages settings

---

<div align="center">

**Ready to deploy?**

Just push to GitHub and enable Pages in repository settings!

</div>
