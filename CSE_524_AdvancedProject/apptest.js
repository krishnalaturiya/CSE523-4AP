var evtSrc   = null;
var finished = false;

function startRun() {
  var btn    = document.getElementById('run-btn');
  var logBox = document.getElementById('log-box');
  var resBox = document.getElementById('result-box');
  var status = document.getElementById('status');

  finished = false;
  btn.disabled     = true;
  btn.textContent  = 'Running…';
  logBox.textContent = '';
  resBox.style.display = 'none';
  resBox.textContent   = '';
  resBox.className     = '';
  status.textContent   = 'Connecting to server…';

  // Close any leftover connection from a previous run
  if (evtSrc) {
    evtSrc.close();
    evtSrc = null;
  }

  // Open the SSE stream — browser auto-reconnects unless we explicitly close it
  evtSrc = new EventSource('/run-gesture-stream');

  // 'log' events: append each line to the live log box
  evtSrc.addEventListener('log', function (e) {
    logBox.textContent += e.data + '\n';
    logBox.scrollTop = logBox.scrollHeight;
  });

  // 'result' event: display the final JSON and reset the UI
  evtSrc.addEventListener('result', function (e) {
    finished = true;

    var data = JSON.parse(e.data);

    resBox.style.display = 'block';
    resBox.textContent   = JSON.stringify(data, null, 2);
    resBox.className     = data.success ? 'ok' : 'err';

    status.textContent = data.success
      ? '✅ Complete — return code ' + data.return_code
      : '❌ Failed — ' + (data.error_message || 'see log above');

    evtSrc.close();
    evtSrc = null;
    btn.disabled    = false;
    btn.textContent = 'Run Gesture Analysis';
  });

  // onerror fires on both real errors and on the normal close after a 'result'
  evtSrc.onerror = function () {
    if (finished) return; // expected close triggered by the 'result' handler

    status.textContent   = '❌ Connection error — check server terminal';
    logBox.textContent  += '\n[ERROR] SSE connection closed unexpectedly.\n';
    evtSrc.close();
    evtSrc = null;
    btn.disabled    = false;
    btn.textContent = 'Run Gesture Analysis';
  };

  status.textContent = 'Running…';
}
