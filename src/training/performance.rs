use std::{
    collections::VecDeque,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use crate::mcts::policy::NetworkTiming;

use super::{accelerator::BatchSummary, self_play::GameTiming};

pub struct UpdateTiming {
    pub total: Duration,
    pub sample: Duration,
    pub forward: Duration,
    pub loss_metrics: Duration,
    pub backward: Duration,
    pub optimizer: Duration,
    pub logging: Duration,
}

pub struct TimingLogger {
    games: BufWriter<File>,
    moves: BufWriter<File>,
    updates: BufWriter<File>,
    iterations: BufWriter<File>,
    evaluations: BufWriter<File>,
    accelerator: BufWriter<File>,
    html_path: PathBuf,
    recent_games: VecDeque<String>,
    recent_updates: VecDeque<String>,
    last_network: String,
    last_iteration: String,
    last_evaluation: String,
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
            games: csv(directory, "timing_games.csv", "iteration,game,moves,total_ms,search_ms,search_evaluation_wait_ms,tree_ms,simulations,simulations_per_second,evaluation_requests,evaluation_wait_total_ms")?,
            moves: csv(directory, "timing_moves.csv", "iteration,game,move,total_ms,search_ms,search_evaluation_wait_ms,tree_ms,simulations,evaluation_requests")?,
            updates: csv(directory, "timing_updates.csv", "step,total_ms,sample_ms,forward_ms,loss_metrics_ms,backward_ms,optimizer_ms,logging_ms")?,
            iterations: csv(directory, "timing_iterations.csv", "iteration,games,positions,total_ms,self_play_ms,updates_ms,evaluation_ms,games_per_second,positions_per_second")?,
            evaluations: csv(directory, "timing_evaluations.csv", "step,games,total_ms,games_per_second")?,
            accelerator: csv(directory, "timing_accelerator.csv", "iteration,requests,batches,largest_batch,model_total_ms,model_input_ms,model_forward_ms,model_readback_ms")?,
            html_path: directory.join("performance.html"),
            recent_games: VecDeque::new(),
            recent_updates: VecDeque::new(),
            last_network: "No self-play evaluation completed yet".to_string(),
            last_iteration: "No completed iteration yet".to_string(),
            last_evaluation: "No evaluation yet".to_string(),
        };
        logger.write_html()?;
        Ok(logger)
    }

    pub fn game(&mut self, iteration: usize, game: usize, timing: &GameTiming, simulations_per_move: u32) -> io::Result<()> {
        let search: Duration = timing.moves.iter().map(|m| m.search).sum();
        let search_wait: Duration = timing.moves.iter().map(|m| m.search_network).sum();
        let tree = search.saturating_sub(search_wait);
        let simulations = timing.moves.len() as u64 * simulations_per_move as u64;
        let simulations_per_second = simulations as f64 / search.as_secs_f64();

        for (index, movement) in timing.moves.iter().enumerate() {
            writeln!(self.moves, "{iteration},{game},{},{:.3},{:.3},{:.3},{:.3},{},{}",
                index + 1, ms(movement.total), ms(movement.search), ms(movement.search_network),
                ms(movement.search.saturating_sub(movement.search_network)),
                simulations_per_move, movement.network_calls)?;
        }
        self.moves.flush()?;
        writeln!(self.games, "{iteration},{game},{},{:.3},{:.3},{:.3},{:.3},{simulations},{simulations_per_second:.2},{},{:.3}",
            timing.moves.len(), ms(timing.total), ms(search), ms(search_wait), ms(tree),
            timing.network.calls, ms(timing.network.total))?;
        self.games.flush()?;

        self.recent_games.push_back(format!(
            "<tr><td>{iteration}.{game}</td><td>{}</td><td>{:.0}</td><td>{:.0}</td><td>{:.0}</td><td>{:.0}</td><td>{}</td><td>{simulations_per_second:.0}</td></tr>",
            timing.moves.len(), ms(timing.total), ms(search), ms(search_wait), ms(tree), timing.network.calls,
        ));
        if self.recent_games.len() > 20 {
            self.recent_games.pop_front();
        }
        self.write_html()?;
        println!("self-play game {iteration}.{game}: {} moves, {:.1}s, {:.0} sims/s, {} evaluation requests (search {:.1}s, reply wait in search {:.1}s)",
            timing.moves.len(), timing.total.as_secs_f64(), simulations_per_second,
            timing.network.calls, search.as_secs_f64(), search_wait.as_secs_f64());
        Ok(())
    }

    pub fn accelerator(&mut self, iteration: usize, summary: &BatchSummary, model: NetworkTiming) -> io::Result<()> {
        writeln!(self.accelerator, "{iteration},{},{},{},{:.3},{:.3},{:.3},{:.3}",
            summary.requests, summary.batches, summary.largest_batch, ms(model.total),
            ms(model.input), ms(model.forward), ms(model.readback))?;
        self.accelerator.flush()?;
        self.last_network = format!("Iteration {iteration}: {} requests in {} batches (largest {}); model input {:.1} ms, forward {:.1} ms, readback {:.1} ms, total {:.1} ms",
            summary.requests, summary.batches, summary.largest_batch, ms(model.input),
            ms(model.forward), ms(model.readback), ms(model.total));
        self.write_html()
    }

    pub fn update(&mut self, step: usize, timing: UpdateTiming) -> io::Result<()> {
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

    pub fn evaluation(&mut self, step: usize, games: usize, elapsed: Duration) -> io::Result<()> {
        let rate = games as f64 / elapsed.as_secs_f64();
        writeln!(self.evaluations, "{step},{games},{:.3},{rate:.3}", ms(elapsed))?;
        self.evaluations.flush()?;
        self.last_evaluation = format!("Step {step}: {games} games in {:.1}s ({rate:.2} games/s)", elapsed.as_secs_f64());
        self.write_html()
    }

    pub fn iteration(&mut self, iteration: usize, games: usize, positions: usize, total: Duration,
        self_play: Duration, updates: Duration, evaluation: Duration) -> io::Result<()> {
        let game_rate = games as f64 / self_play.as_secs_f64();
        let position_rate = positions as f64 / self_play.as_secs_f64();
        writeln!(self.iterations, "{iteration},{games},{positions},{:.3},{:.3},{:.3},{:.3},{game_rate:.3},{position_rate:.3}",
            ms(total), ms(self_play), ms(updates), ms(evaluation))?;
        self.iterations.flush()?;
        self.last_iteration = format!("Iteration {iteration}: total {:.1}s; self-play {:.1}s ({game_rate:.2} games/s, {position_rate:.1} positions/s); updates {:.1}s; evaluation {:.1}s",
            total.as_secs_f64(), self_play.as_secs_f64(), updates.as_secs_f64(), evaluation.as_secs_f64());
        self.write_html()
    }

    fn write_html(&self) -> io::Result<()> {
        let mut html = String::from("<!doctype html><html><head><meta charset=\"utf-8\"><meta http-equiv=\"refresh\" content=\"5\"><title>Connect 4 performance</title><style>body{font-family:system-ui,sans-serif;background:#101827;color:#e5eaf3;margin:24px}section{background:#1a2537;padding:16px;border-radius:10px;margin:16px 0;overflow-x:auto}table{border-collapse:collapse;width:100%}th,td{text-align:right;padding:6px 10px;border-bottom:1px solid #344155}th:first-child,td:first-child{text-align:left}small{color:#b9c4d6}</style></head><body><h1>Connect 4 performance</h1><p>Refreshes every 5 seconds. Durations are wall-clock milliseconds; evaluation wait is included in search time.</p>");
        html.push_str(&format!("<section><h2>Latest self-play evaluator</h2><p>{}</p></section><section><h2>Latest iteration</h2><p>{}</p></section><section><h2>Latest evaluation</h2><p>{}</p></section>",
            self.last_network, self.last_iteration, self.last_evaluation));
        html.push_str("<section><h2>Recent self-play games</h2><table><tr><th>Game</th><th>Moves</th><th>Total ms</th><th>Search ms</th><th>Evaluation wait in search ms</th><th>Tree/other ms</th><th>Evaluation requests</th><th>Sims/s</th></tr>");
        for row in self.recent_games.iter().rev() { html.push_str(row); }
        html.push_str("</table></section><section><h2>Recent training updates</h2><table><tr><th>Step</th><th>Total ms</th><th>Sample ms</th><th>Forward ms</th><th>Loss/metrics ms</th><th>Backward ms</th><th>Optimizer ms</th><th>Logging ms</th></tr>");
        for row in self.recent_updates.iter().rev() { html.push_str(row); }
        html.push_str("</table></section><small>GPU work is asynchronous: forward/optimizer are host-side timings. Readbacks and metric extraction synchronize the device. See timing_*.csv for per-move and network input/forward/readback details.</small></body></html>");
        fs::write(&self.html_path, html)
    }
}
