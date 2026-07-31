document.addEventListener("DOMContentLoaded", function() {
    fetch('https://librosa.org/web-staging/versions.json')
        .then(response => response.json())
        .then(data => {
            const currentPath = window.location.pathname;
            const preferred = data.find(item => item.preferred);
            
            // 1. Evaluate State
            const isDev = currentPath.includes('/dev/');
            const isPreferred = preferred && currentPath.includes(`/${preferred.version}/`);
            
            // 2. Inject Warning Banner
            if (preferred && !isDev && !isPreferred) {
                const banner = document.createElement('div');
                banner.innerHTML = `<strong>Warning:</strong> You are viewing an older version of this documentation. <a href="${preferred.url}" style="color: #721c24; text-decoration: underline;">Switch to version ${preferred.version}</a>.`;
                banner.style.cssText = 'background-color: #f8d7da; color: #721c24; padding: 10px; text-align: center; border-bottom: 1px solid #f5c6cb; font-family: sans-serif;';
                document.body.insertBefore(banner, document.body.firstChild);
            }

            // 3. Construct RTD Dropdown (Previous logic)
            let linksHtml = data.map(item => {
                const isCurrent = currentPath.includes(`/${item.version}/`);
                return `<dd><a href="${item.url}">${item.version} ${isCurrent ? '(current)' : ''}</a></dd>`;
            }).join('');

            const rtdMenuHtml = `
                <div class="rst-versions" data-toggle="rst-versions" role="note" aria-label="versions" style="display:block;">
                    <span class="rst-current-version" data-toggle="rst-current-version">
                        <span class="fa fa-book"> Versions</span>
                        <span class="fa fa-caret-down"></span>
                    </span>
                    <div class="rst-other-versions">
                        <dl><dt>Versions</dt>${linksHtml}</dl>
                    </div>
                </div>
            `;
            document.body.insertAdjacentHTML('beforeend', rtdMenuHtml);
            
// 4. Bind event listener strictly to the new menu
            const currentVersionBtn = newMenu.querySelector('.rst-current-version');
            const otherVersions = newMenu.querySelector('.rst-other-versions');
            
            // Force the initial hidden state just in case legacy CSS doesn't apply it
            if (otherVersions) {
                otherVersions.style.display = 'none';
            }

            if (currentVersionBtn) {
                currentVersionBtn.addEventListener('click', (e) => {
                    // Prevent default anchor behavior
                    e.preventDefault(); 
                    
                    // CRITICAL: Stop the click from bubbling to document and triggering native RTD jQuery
                    e.stopPropagation(); 
                    
                    // Toggle the native class for UI changes (like the caret arrow)
                    newMenu.classList.toggle('shift-up');
                    
                    // Force the display state inline, completely bypassing CSS failures
                    if (otherVersions.style.display === 'none' || otherVersions.style.display === '') {
                        otherVersions.style.display = 'block';
                    } else {
                        otherVersions.style.display = 'none';
                    }
                });
            }
        })
        .catch(error => console.error('Failed to load versions.json:', error));
});
