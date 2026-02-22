fetch('/api/health')
    .then(r => r.json())
    .then(data => {
        const el = document.getElementById('status');
        if (data && data.ok) {
            el.className = 'ok';
            el.querySelector('.text').textContent = 'API connected';
        } else {
            throw new Error('not ok');
        }
    })
    .catch(() => {
        const el = document.getElementById('status');
        el.className = 'err';
        el.querySelector('.text').textContent = 'API unreachable';
    });
