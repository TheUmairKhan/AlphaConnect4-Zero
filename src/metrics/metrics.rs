use std::{
    collections::VecDeque,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

use burn::prelude::*;

use crate::training::evaluation::arena::MatchSummary;

pub struct StepMetrics {
    pub step: usize,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub value_mae: f32,
    pub policy_entropy: f32,
}

impl StepMetrics {
    pub fn from_batch<B: Backend>(
        step: usize,
        policy_loss: &Tensor<B, 1>,
        value_loss: &Tensor<B, 1>,
        log_network_policy: Tensor<B, 2>,
        value_error: Tensor<B, 2>,
    ) -> Self {
        let policy_entropy =
            -(log_network_policy.clone().exp() * log_network_policy).sum_dim(1).mean();
        let value_mae = value_error.abs().mean();

        let values: Vec<f32> = Tensor::cat(
            vec![policy_loss.clone(), value_loss.clone(), value_mae, policy_entropy],
            0,
        )
        .into_data()
        .iter::<f32>()
        .collect();

        Self {
            step,
            policy_loss: values[0],
            value_loss: values[1],
            value_mae: values[2],
            policy_entropy: values[3],
        }
    }
}

struct Evaluation {
    step: usize,
    summary: MatchSummary,
}

struct Series {
    name: String,
    color: &'static str,
    points: Vec<(f32, f32)>,
}

const DEPTH_COLORS: [&str; 8] = ["#60a5fa", "#fb923c", "#34d399", "#f472b6", "#a78bfa", "#facc15", "#22d3ee", "#f87171"];

pub struct MetricsLogger {
    csv: BufWriter<File>,
    evaluation_csv: BufWriter<File>,
    html_path: PathBuf,
    recent: VecDeque<StepMetrics>,
    evaluations: Vec<Evaluation>,
}

impl MetricsLogger {
    pub fn new(directory: &Path) -> io::Result<Self> {
        let mut csv = BufWriter::new(File::create(directory.join("metrics.csv"))?);
        writeln!(csv, "step,policy_loss,value_loss,value_mae,policy_entropy")?;
        let mut evaluation_csv = BufWriter::new(File::create(directory.join("evaluation.csv"))?);
        writeln!(
            evaluation_csv,
            "step,depth,games,wins,losses,draws,win_pct,loss_pct,draw_pct,score_rate,elo,first_player_score_rate,second_player_score_rate"
        )?;
        Ok(Self {
            csv,
            evaluation_csv,
            html_path: directory.join("metrics.html"),
            recent: VecDeque::new(),
            evaluations: Vec::new(),
        })
    }

    pub fn record(&mut self, metrics: StepMetrics) -> io::Result<()> {
        writeln!(
            self.csv,
            "{},{},{},{},{}",
            metrics.step,
            metrics.policy_loss,
            metrics.value_loss,
            metrics.value_mae,
            metrics.policy_entropy,
        )?;
        self.csv.flush()?;
        self.recent.push_back(metrics);
        if self.recent.len() > 1000 {
            self.recent.pop_front();
        }
        fs::write(&self.html_path, self.dashboard())
    }

    pub fn record_evaluation(&mut self, step: usize, summaries: &[MatchSummary]) -> io::Result<()> {
        for &summary in summaries {
            writeln!(
                self.evaluation_csv,
                "{},{},{},{},{},{},{},{},{},{},{},{},{}",
                step,
                summary.depth,
                summary.games(),
                summary.wins,
                summary.losses,
                summary.draws,
                100.0 * summary.win_rate(),
                100.0 * summary.loss_rate(),
                100.0 * summary.draw_rate(),
                summary.score_rate(),
                summary.elo(),
                summary.first_player_score_rate(),
                summary.second_player_score_rate(),
            )?;
            self.evaluations.push(Evaluation { step, summary });
        }
        self.evaluation_csv.flush()?;
        fs::write(&self.html_path, self.dashboard())
    }

    fn dashboard(&self) -> String {
        let mut html = String::from(
            r#"<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>Connect 4 training</title><style>body{font-family:system-ui,sans-serif;background:#101827;color:#e5eaf3;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;margin-bottom:24px}.card{background:#1a2537;border-radius:10px;padding:16px}svg{width:100%;height:auto}h1{font-size:1.5rem}h2{font-size:1rem;margin:0 0 8px}.legend{font-size:.8rem;margin:8px 0;color:#b9c4d6}table{border-collapse:collapse;font-size:.85rem;font-variant-numeric:tabular-nums}th,td{padding:4px 12px;text-align:right;border-bottom:1px solid #2a3a52}th{color:#b9c4d6;font-weight:600}</style></head><body><h1>Connect 4 training</h1>"#,
        );
        if let Some(last) = self.recent.back() {
            html.push_str(&format!("<p>Step {} · Refreshes every 5 seconds · Showing up to 1,000 recent steps</p>", last.step));
        }

        html.push_str("<h2>Training</h2><main>");
        let training = |name: &str, color, value: fn(&StepMetrics) -> f32| Series {
            name: name.to_string(),
            color,
            points: self.recent.iter().map(|m| (m.step as f32, value(m))).collect(),
        };
        html.push_str(&chart("Policy loss", &[training("Policy loss", "#60a5fa", |m| m.policy_loss)], None, false));
        html.push_str(&chart("Value loss", &[training("Value loss", "#fb923c", |m| m.value_loss)], None, false));
        html.push_str(&chart("Value MAE", &[training("MAE", "#34d399", |m| m.value_mae)], None, false));
        html.push_str(&chart("Policy entropy", &[training("Entropy", "#a78bfa", |m| m.policy_entropy)], None, false));
        html.push_str("</main>");

        html.push_str("<h2>Evaluation vs minimax</h2>");
        if self.evaluations.is_empty() {
            html.push_str("<p class=\"legend\">No evaluations yet.</p>");
        } else {
            html.push_str(&self.latest_table());
            html.push_str("<main>");
            let percent = Some((0.0, 100.0));
            let rate = Some((0.0, 1.0));
            html.push_str(&chart("Win %", &self.by_depth(|s| 100.0 * s.win_rate()), percent, true));
            html.push_str(&chart("Loss %", &self.by_depth(|s| 100.0 * s.loss_rate()), percent, true));
            html.push_str(&chart("Draw %", &self.by_depth(|s| 100.0 * s.draw_rate()), percent, true));
            html.push_str(&chart("Score rate", &self.by_depth(MatchSummary::score_rate), rate, true));
            html.push_str(&chart("Elo vs minimax (minimax = 0)", &self.by_depth(MatchSummary::elo), None, true));
            html.push_str(&chart("First-player score rate", &self.by_depth(MatchSummary::first_player_score_rate), rate, true));
            html.push_str(&chart("Second-player score rate", &self.by_depth(MatchSummary::second_player_score_rate), rate, true));
            html.push_str("</main>");
        }
        html.push_str("</body></html>");
        html
    }

    fn depths(&self) -> Vec<u32> {
        let mut depths: Vec<u32> = self.evaluations.iter().map(|e| e.summary.depth).collect();
        depths.sort_unstable();
        depths.dedup();
        depths
    }

    fn by_depth(&self, value: fn(&MatchSummary) -> f32) -> Vec<Series> {
        self.depths().into_iter().enumerate().map(|(index, depth)| Series {
            name: format!("Depth {depth}"),
            color: DEPTH_COLORS[index % DEPTH_COLORS.len()],
            points: self.evaluations.iter()
                .filter(|e| e.summary.depth == depth)
                .map(|e| (e.step as f32, value(&e.summary)))
                .collect(),
        }).collect()
    }

    fn latest_table(&self) -> String {
        let step = self.evaluations.last().map_or(0, |e| e.step);
        let mut html = format!(
            "<div class=\"card\" style=\"margin-bottom:16px\"><h2>Latest checkpoint: step {step}</h2><table><tr><th>Depth</th><th>Games</th><th>Win %</th><th>Loss %</th><th>Draw %</th><th>Score rate</th><th>Elo</th><th>First-player score</th><th>Second-player score</th></tr>"
        );
        for e in self.evaluations.iter().filter(|e| e.step == step) {
            let s = &e.summary;
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{:.1}</td><td>{:.1}</td><td>{:.1}</td><td>{:.3}</td><td>{:+.0}</td><td>{:.3}</td><td>{:.3}</td></tr>",
                s.depth, s.games(), 100.0 * s.win_rate(), 100.0 * s.loss_rate(), 100.0 * s.draw_rate(),
                s.score_rate(), s.elo(), s.first_player_score_rate(), s.second_player_score_rate(),
            ));
        }
        html.push_str("</table></div>");
        html
    }
}

fn chart(title: &str, series: &[Series], y_range: Option<(f32, f32)>, markers: bool) -> String {
    let points = || series.iter().flat_map(|s| s.points.iter().copied());
    let (x_min, x_max) = points().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), (x, _)| (lo.min(x), hi.max(x)));
    let (y_min, y_max) = y_range.unwrap_or_else(|| {
        let (lo, hi) = points().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), (_, y)| (lo.min(y), hi.max(y)));
        if lo > hi { (0.0, 1.0) } else { (lo.min(0.0), if hi - lo.min(0.0) < 1e-6 { lo.min(0.0) + 1.0 } else { hi }) }
    });
    let x_span = (x_max - x_min).max(1e-6);
    let to_x = |x: f32| if x_max > x_min { 45.0 + 575.0 * (x - x_min) / x_span } else { 45.0 };
    let to_y = |y: f32| 190.0 - 170.0 * ((y - y_min) / (y_max - y_min)).clamp(0.0, 1.0);

    let mut html = format!(
        "<div class=\"card\"><h2>{title}</h2><svg viewBox=\"0 0 640 220\"><path d=\"M45 20 V190 H620\" fill=\"none\" stroke=\"#64748b\"/><text x=\"3\" y=\"25\" fill=\"#b9c4d6\" font-size=\"12\">{y_max:.2}</text><text x=\"3\" y=\"190\" fill=\"#b9c4d6\" font-size=\"12\">{y_min:.2}</text>"
    );
    if y_min < 0.0 && y_max > 0.0 {
        html.push_str(&format!("<path d=\"M45 {0:.1} H620\" stroke=\"#64748b\" stroke-dasharray=\"4 4\"/>", to_y(0.0)));
    }
    if x_min <= x_max {
        html.push_str(&format!(
            "<text x=\"45\" y=\"210\" fill=\"#b9c4d6\" font-size=\"12\">step {x_min}</text><text x=\"550\" y=\"210\" fill=\"#b9c4d6\" font-size=\"12\">{x_max}</text>"
        ));
    }
    for s in series {
        html.push_str(&format!("<polyline fill=\"none\" stroke=\"{}\" stroke-width=\"2\" points=\"", s.color));
        for &(x, y) in &s.points {
            html.push_str(&format!("{:.1},{:.1} ", to_x(x), to_y(y)));
        }
        html.push_str("\"/>");
        let marked: &[(f32, f32)] = if markers { &s.points } else { s.points.last().map_or(&[], std::slice::from_ref) };
        for &(x, y) in marked {
            html.push_str(&format!("<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"3\" fill=\"{}\"/>", to_x(x), to_y(y), s.color));
        }
    }
    html.push_str("</svg><div class=\"legend\">");
    for s in series {
        if let Some(&(_, y)) = s.points.last() {
            html.push_str(&format!("<span style=\"color:{}\">● {}: {:.3}</span> &nbsp;", s.color, s.name, y));
        }
    }
    html.push_str("</div></div>");
    html
}
