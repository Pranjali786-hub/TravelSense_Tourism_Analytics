// Keeps Bootstrap alerts tidy after a short delay on this simple local dashboard.
document.querySelectorAll('.alert').forEach(function (alert) {
  setTimeout(function () { bootstrap.Alert.getOrCreateInstance(alert).close(); }, 4500);
});
