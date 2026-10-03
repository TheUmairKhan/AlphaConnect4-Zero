use std::{
    collections::VecDeque,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use crate::mcts::policy::NetworkTiming;

use crate::training::self_play::self_play::GameTiming;

pub struct UpdateTiming {
    pub total: Duration,
    pub sample: Duration,
    pub forward: Duration,
    pub loss_metrics: Duration,
    pub backward: Duration,
    pub optimizer: Duration,
    pub logging: Duration,
}

const RATE_WINDOW_SECS: f64 = 30.0;

struct GameSample {
    t: f64,
    positions: usize,
    eval_wait: Duration,
    eval_calls: usize,
}

fn prune<T>(window: &mut VecDeque<T>, now: f64, time_of: impl Fn(&T) -> f64) {
    while window.len() > 1 && now - time_of(window.front().unwrap()) > RATE_WINDOW_SECS {
        window.pop_front();
    }
}

pub struct TimingLogger {
    games: BufWriter<File>,
    moves: BufWriter<File>,
    updates: BufWriter<File>,
    run: BufWriter<File>,
    accelerator: BufWriter<File>,
    html_path: PathBuf,
    recent_games: VecDeque<String>,
    recent_updates: VecDeque<String>,
    started: Instant,
    batches: usize,
    requests: usize,
    game_window: VecDeque<GameSample>,
    eval_window: VecDeque<(f64, usize)>,
    update_window: VecDeque<(f64, Duration)>,
    model_timing: NetworkTiming,
    last_network: String,
    last_run: String,
}

fn csv(path: &Path, name: &str, header: &str) -> io::Result<BufWriter<File>> {
    let mut writer = BufWriter::new(File::create(path.join(name))?);
    writeln!(writer, "{header}")?;
    writer.flush()?;
    Ok(writer)
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

impl TimingLogger {
    pub fn new(directory: &Path) -> io::Result<Self> {
        let logger = Self {
            games: csv(directory, "timing_games.csv", "game,moves,total_ms,search_ms,search_evaluation_wait_ms,tree_ms,simulations,simulations_per_second,evaluation_requests,evaluation_wait_total_ms")?,
            moves: csv(directory, "timing_moves.csv", "game,move,total_ms,search_ms,search_evaluation_wait_ms,tree_ms,simulations,evaluation_requests")?,
            updates: csv(directory, "timing_updates.csv", "step,total_ms,sample_ms,forward_ms,loss_metrics_ms,backward_ms,optimizer_ms,logging_ms")?,
            run: csv(directory, "timing_run.csv", "updates,games,positions,total_ms,games_per_second,positions_per_second")?,
            accelerator: csv(directory, "timing_accelerator.csv", "batch,model_step,requests,total_requests,model_total_ms,model_input_ms,model_forward_ms,model_readback_ms")?,
            html_path: directory.join("performance.html"),
            recent_games: VecDeque::new(),
            recent_updates: VecDeque::new(),
            started: Instant::now(),
            batches: 0,
            requests: 0,
            game_window: VecDeque::new(),
            eval_window: VecDeque::new(),
            update_window: VecDeque::new(),
            model_timing: NetworkTiming::default(),
            last_network: "No self-play evaluation completed yet".to_string(),
            last_run: "Training in progress".to_string(),
        };
        logger.write_html()?;
        Ok(logger)
    }

    fn now(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    pub fn game(&mut self, game: usize, timing: &GameTiming, simulations_per_move: u32, positions: usize) -> io::Result<()> {
        let now = self.now();
        self.game_window.push_back(GameSample {
            t: now,
            positions,
            eval_wait: timing.network.total,
            eval_calls: timing.network.calls,
        });
        prune(&mut self.game_window, now, |sample| sample.t);
        let search: Duration = timing.moves.iter().map(|m| m.search).sum();
        let search_wait: Duration = timing.moves.iter().map(|m| m.search_network).sum();
        let tree = search.saturating_sub(search_wait);
        let simulations = timing.moves.len() as u64 * simulations_per_move as u64;
        let simulations_per_second = simulations as f64 / search.as_secs_f64();

        for (index, movement) in timing.moves.iter().enumerate() {
            writeln!(self.moves, "{game},{},{:.3},{:.3},{:.3},{:.3},{},{}",
                index + 1, ms(movement.total), ms(movement.search), ms(movement.search_network),
                ms(movement.search.saturating_sub(movement.search_network)),
                simulations_per_move, movement.network_calls)?;
        }
        self.moves.flush()?;
        writeln!(self.games, "{game},{},{:.3},{:.3},{:.3},{:.3},{simulations},{simulations_per_second:.2},{},{:.3}",
            timing.moves.len(), ms(timing.total), ms(search), ms(search_wait), ms(tree),
            timing.network.calls, ms(timing.network.total))?;
        self.games.flush()?;

        self.recent_games.push_back(format!(
            "<tr><td>{game}</td><td>{}</td><td>{:.0}</td><td>{:.0}</td><td>{:.0}</td><td>{:.0}</td><td>{}</td><td>{simulations_per_second:.0}</td></tr>",
            timing.moves.len(), ms(timing.total), ms(search), ms(search_wait), ms(tree), timing.network.calls,
        ));
        if self.recent_games.len() > 20 {
            self.recent_games.pop_front();
        }
        self.write_html()?;
        println!("self-play game {game}: {} moves, {:.1}s, {:.0} sims/s, {} evaluation requests (search {:.1}s, reply wait in search {:.1}s)",
            timing.moves.len(), timing.total.as_secs_f64(), simulations_per_second,
            timing.network.calls, search.as_secs_f64(), search_wait.as_secs_f64());
        Ok(())
    }

    pub fn accelerator_batch(&mut self, model_step: usize, requests: usize, timing: NetworkTiming) -> io::Result<()> {
        let now = self.now();
        self.eval_window.push_back((now, requests));
        prune(&mut self.eval_window, now, |sample| sample.0);
        self.batches += 1;
        self.requests += requests;
        self.model_timing.input += timing.input;
        self.model_timing.forward += timing.forward;
        self.model_timing.readback += timing.readback;
        self.model_timing.total += timing.total;
        writeln!(self.accelerator, "{},{model_step},{requests},{},{:.3},{:.3},{:.3},{:.3}",
            self.batches, self.requests, ms(timing.total), ms(timing.input),
            ms(timing.forward), ms(timing.readback))?;
        self.accelerator.flush()?;
        self.last_network = format!("{} requests in {} batches; latest batch {} requests using model step {model_step}; total model input {:.1} ms, forward {:.1} ms, readback {:.1} ms, total {:.1} ms",
            self.requests, self.batches, requests, ms(self.model_timing.input),
            ms(self.model_timing.forward), ms(self.model_timing.readback), ms(self.model_timing.total));
        Ok(())
    }

    pub fn update(&mut self, step: usize, timing: UpdateTiming) -> io::Result<()> {
        let now = self.now();
        self.update_window.push_back((now, timing.total));
        prune(&mut self.update_window, now, |sample| sample.0);
        writeln!(self.updates, "{step},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            ms(timing.total), ms(timing.sample), ms(timing.forward), ms(timing.loss_metrics),
            ms(timing.backward), ms(timing.optimizer), ms(timing.logging))?;
        self.updates.flush()?;
        self.recent_updates.push_back(format!(
            "<tr><td>{step}</td><td>{:.0}</td><td>{:.0}</td><td>{:.0}</td><td>{:.0}</td><td>{:.0}</td><td>{:.0}</td><td>{:.0}</td></tr>",
            ms(timing.total), ms(timing.sample), ms(timing.forward), ms(timing.loss_metrics),
            ms(timing.backward), ms(timing.optimizer), ms(timing.logging),
        ));
        if self.recent_updates.len() > 20 {
            self.recent_updates.pop_front();
        }
        self.write_html()
    }

    pub fn finish(&mut self, updates: usize, games: usize, positions: usize, total: Duration) -> io::Result<()> {
        let game_rate = games as f64 / total.as_secs_f64();
        let position_rate = positions as f64 / total.as_secs_f64();
        writeln!(self.run, "{updates},{games},{positions},{:.3},{game_rate:.3},{position_rate:.3}", ms(total))?;
        self.run.flush()?;
        self.last_run = format!("{updates} updates, {games} games, {positions} positions in {:.1}s ({game_rate:.2} games/s)", total.as_secs_f64());
        self.write_html()
    }

    fn write_html(&self) -> io::Result<()> {
        let now = self.now();

        let (games_per_second, positions_per_second) = if self.game_window.len() >= 2 {
            let span = (now - self.game_window.front().unwrap().t).max(f64::MIN_POSITIVE);
            let positions: usize = self.game_window.iter().map(|sample| sample.positions).sum();
            (self.game_window.len() as f64 / span, positions as f64 / span)
        } else {
            (0.0, 0.0)
        };

        let evals_per_second = if self.eval_window.len() >= 2 {
            let span = (now - self.eval_window.front().unwrap().0).max(f64::MIN_POSITIVE);
            let requests: usize = self.eval_window.iter().map(|(_, requests)| requests).sum();
            requests as f64 / span
        } else {
            0.0
        };

        let avg_eval_latency_ms = {
            let wait: Duration = self.game_window.iter().map(|sample| sample.eval_wait).sum();
            let calls: usize = self.game_window.iter().map(|sample| sample.eval_calls).sum();
            if calls > 0 { ms(wait) / calls as f64 } else { 0.0 }
        };

        let avg_update_ms = if !self.update_window.is_empty() {
            let total: Duration = self.update_window.iter().map(|(_, total)| *total).sum();
            ms(total) / self.update_window.len() as f64
        } else {
            0.0
        };

        let model_total = self.model_timing.total.as_secs_f64();
        let (input_pct, forward_pct, readback_pct) = if model_total > 0.0 {
            (
                self.model_timing.input.as_secs_f64() / model_total * 100.0,
                self.model_timing.forward.as_secs_f64() / model_total * 100.0,
                self.model_timing.readback.as_secs_f64() / model_total * 100.0,
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        let mut html = String::from("<!doctype html><html><head><meta charset=\"utf-8\"><meta http-equiv=\"refresh\" content=\"5\"><title>Connect 4 performance</title><style>body{font-family:system-ui,sans-serif;background:#101827;color:#e5eaf3;margin:24px}section{background:#1a2537;padding:16px;border-radius:10px;margin:16px 0;overflow-x:auto}table{border-collapse:collapse;width:100%}th,td{text-align:right;padding:6px 10px;border-bottom:1px solid #344155}th:first-child,td:first-child{text-align:left}small{color:#b9c4d6}</style></head><body><h1>Connect 4 performance</h1><p>Refreshes every 5 seconds. Durations are wall-clock milliseconds; evaluation wait is included in search time.</p>");
        html.push_str(&format!(
            "<section><h2>Rates</h2><p>{games_per_second:.2} games/s &middot; {positions_per_second:.1} positions/s &middot; {evals_per_second:.1} evals/s &middot; {avg_eval_latency_ms:.2} ms avg evaluation latency &middot; {avg_update_ms:.1} ms/update</p><p><small>Rolling average over the last {RATE_WINDOW_SECS:.0}s (or less, early in a run).</small></p></section>"
        ));
        html.push_str(&format!(
            "<section><h2>Self-play evaluator</h2><p>{}</p><p>GPU time split: input {input_pct:.1}% &middot; forward {forward_pct:.1}% &middot; readback {readback_pct:.1}%</p></section><section><h2>Training run</h2><p>{}</p></section>",
            self.last_network, self.last_run));
        html.push_str("<section><h2>Recent self-play games</h2><table><tr><th>Game</th><th>Moves</th><th>Total ms</th><th>Search ms</th><th>Evaluation wait in search ms</th><th>Tree/other ms</th><th>Evaluation requests</th><th>Sims/s</th></tr>");
        for row in self.recent_games.iter().rev() { html.push_str(row); }
        html.push_str("</table></section><section><h2>Recent training updates</h2><table><tr><th>Step</th><th>Total ms</th><th>Sample ms</th><th>Forward ms</th><th>Loss/metrics ms</th><th>Backward ms</th><th>Optimizer ms</th><th>Logging ms</th></tr>");
        for row in self.recent_updates.iter().rev() { html.push_str(row); }
        html.push_str("</table></section><small>GPU work is asynchronous. The self-play evaluator's forward timing syncs the device, so its input/forward/readback split is accurate. Training-update forward/backward/optimizer timings do not sync and only measure host-side kernel enqueue time. See timing_*.csv for per-move and network input/forward/readback details.</small></body></html>");
        fs::write(&self.html_path, html)
    }
}
