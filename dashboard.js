// dashboard.js
document.addEventListener("DOMContentLoaded", () => {
  const trainBtn = document.getElementById("trainBtn");
  const trainProgress = document.getElementById("trainProgress");
  const trainMsg = document.getElementById("trainMsg");

  async function pollStatus() {
    try {
      const res = await fetch("/train_status");
      const data = await res.json();
      trainProgress.style.width = data.progress + "%";
      trainProgress.innerText = data.progress + "%";
      trainMsg.innerText = data.message || "";
      return data;
    } catch (e) {
      console.error(e);
      return null;
    }
  }

  trainBtn.addEventListener("click", async () => {
    trainBtn.disabled = true;
    const start = await fetch("/train_model");
    if (!start.ok && start.status !== 202) {
      alert("Failed to start training");
      trainBtn.disabled = false;
      return;
    }
    trainMsg.innerText = "Training started...";
    trainProgress.style.width = "10%";
    trainProgress.innerText = "10%";
    
    let lastProgress = 0;
    const t = setInterval(async () => {
      const s = await pollStatus();
      if (s) {
        lastProgress = s.progress;
        console.log("Training status:", s);
        
        if (s.progress >= 100 || (s.message && s.message.includes("complete"))) {
          clearInterval(t);
          trainBtn.disabled = false;
          trainProgress.style.width = "100%";
          trainProgress.innerText = "100%";
          setTimeout(() => {
            alert("Training Completed Successfully!");
          }, 500);
          return;
        }
      }
    }, 1000);
  });

  // Chart initial render & update every 10s
  let chart = null;
  async function updateChart() {
    const res = await fetch("/attendance_stats");
    const data = await res.json();
    const ctx = document.getElementById("attendanceChart").getContext("2d");
    if (!chart) {
      chart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: data.dates,
          datasets: [{ label: "Attendance", data: data.counts, backgroundColor: "rgba(59,130,246,0.7)" }]
        },
        options: { responsive: true, maintainAspectRatio: false }
      });
    } else {
      chart.data.labels = data.dates;
      chart.data.datasets[0].data = data.counts;
      chart.update();
    }
  }
  updateChart();
  setInterval(updateChart, 10000);
});
