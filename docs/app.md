# Smile Board (web app)

A local web app for the spline + bells model on the saved SPX and RUT slices: the
volatility smile, the option board, and the bell buttons along the bottom.

```bash
.venv/Scripts/python app/server.py
```

Then open <http://127.0.0.1:8050> (`--port 9000` for another port). Python
does all the maths; the page only draws and sends button values. It uses only
the standard library HTTP server, bound to 127.0.0.1, so it is not reachable
from other machines.

## Screen

**Top bar**
- **Slice**: which saved index and expiry, `spx_…` or `rut_…` (defaults to the SPX 2026-10-16 monthly).
- **Knots**: interior knots of the spline base (3 default, 0–30). Changing it
  refits the base and the bells.
- **k**: bell width multiplier (0.4 default).
- **Delta**: place the bells at the *final smile's* delta (default) or the
  *spline base smile's*. See [models.md](models.md#final-smile-delta-vs-base-smile-delta).
- **Step**: how far one click moves a button (0.1 / 0.25 / 0.5 / 1 vol point).
- **Refit**: fit a fresh spline and fresh bells to the market mids, and clear
  your offsets.
- **Auto-fit bells**: the automatic fitter, run once: keep the spline, refit
  only the bells to the mids starting from the current fitted values, keep
  your offsets on top.
- **Auto**: run the automatic fitter whenever k, Delta or Stiffness change
  (later: on each new snapshot). Off: those changes only redraw.
- **Stiffness**: automatic fitter only. How strongly the bells stay near their
  previous fitted values (0 = pure fit to the mids). 1 means moving a button
  0.1 vol points costs as much as one quote missing by 0.1 vol points.
- **Clear offsets**: offsets to 0, fitted bells kept.
- **Zero buttons**: fitted bells and offsets to 0, keeping the spline base.
- **Show SVI**: draw SVI fitted to the same mids, for comparison.
- **Full strike range**: by default the chart shows the 1Δ put to 1Δ call
  nodes plus a margin; tick to start from every quote.
- Stats: forward, tau, quote time, RMSE against the mids in vol points
  (spline alone, the drawn smile with your offsets, SVI), and how the last
  bell fit went: its time and solver, e.g. `bells fit 2.8 ms (fast)`, or
  `scipy after fast: <reason>` when the fast solver handed over
  ([calibration.md](calibration.md#fast-solver-default)).
- **Stale banner** (amber), after an automatic fit: the spline looks stale
  and a Refit is due. Shown when the bells' fit error (without offsets) is
  above 1.5× the error just after the last Refit (made with the k and Delta
  in use then, so changing them can trip it), a fitted button is beyond
  2 vol points, or the fast solver met an ambiguous vol. Refit clears it.

**Smile chart (left)**
- Black dots: market mid implied vols. Grey bars: bid to ask vols.
- Blue dashed: spline base. Orange: spline + bells. Solid green (when shown):
  SVI. Grey vertical line: forward.
- **+ / −** (top right): zoom in or out around the middle of the view.
  **Drag** the chart left or right to move along the strikes. The vol axis
  fits whatever is in view. **Reset view** (or changing slice or Full strike
  range) goes back to the default view. The view stays within the quoted
  strikes.
- Orange ticks on the x axis: where each bell node sits on the current smile.
- Hover for the nearest quote's mid, bid/ask and model vol and delta.

**Option board (right)** — one row per out-of-the-money quote:
Type, Strike, Delta (model), Bid, Ask, Mid, Model price (Black-76 at the model
vol), IV bid / mid / ask, Model IV, Model − mid (vol points), Volume, Open
interest. The row nearest the forward is highlighted; a model IV outside the
bid/ask is shaded red.

**Buttons (bottom)** — the 11 bells, 1Δ put … ATM … 1Δ call, values in vol points.
Each shows **fitted + offset**, with your offset underneath when it is not 0.

- **Fitted**: set by Refit or the automatic fitter, always a fit to the mids.
- **Offset**: yours. Clicks change only the offset. The automatic fitter never
  touches it (it fits the bells to the mids and your offset stays on top), so
  a click is not undone by the next automatic fit. Refit clears it.

| Action | Effect |
|---|---|
| Click (left or right mouse button) | raise your offset by one step |
| Ctrl + click (left or right) | lower your offset by one step |

Each press of the mouse button counts once (it is read on mouse-down), so a
Mac Ctrl + click lowers once. The browser's right-click menu is suppressed on
the buttons. Quick repeated clicks add up: each builds on the value shown, not
on the last server reply. Clicks and setting changes are ignored while a fit
(Refit or automatic) is running (the cursor shows busy).

If a change would make some strike's vol ambiguous (see
[models.md](models.md#ambiguous-vols)), the server refuses it, a red banner
explains why, and your offsets return to the last accepted values.

## API

| Method and path | Body / query | Returns |
|---|---|---|
| `GET /api/slices` | | names of the saved slices |
| `GET /api/fit` | `slice`, `knots` (3), `k` (0.4), `delta` (`final`) | Refit: fresh spline and bells fitted to the mids, the smile and board, `fitted` and a `fit` report |
| `POST /api/autofit` | JSON `{slice, base_slice, knots, fitted: {...}, offsets: {...}, k, delta, stiffness, rmse_at_fit (optional: without it the error-ratio test is skipped), stale_ratio (1.5), stale_button (0.02)}` | automatic fitter: base_slice's spline kept, bells refitted to slice's mids from `fitted`; smile drawn with fitted + offsets; `fitted`, `offsets`, `fit`, `stale` (list of reasons) |
| `POST /api/smile` | JSON `{slice, base_slice, knots, buttons: {...}, k, delta}` | the smile and board for those buttons on base_slice's spline (default: slice); buttons left out are 0, knots default 3 |

`base_slice` must be the same index and expiry as `slice` (a later snapshot of
the same expiry keeps the spline fitted to an earlier one, moved with the new
forward). The `fit` report is `{method, handover, converged, ms, rmse}` (plus
`spline_ms` from `/api/fit`); `rmse` is the bells' fit to the mids without
offsets.

The server fits the spline base (per slice and knot count) and SVI (per slice)
once and keeps them, so button clicks only recompute the bells.

Response fields: `slice {name, expiry, as_of, forward, tau}`, `base_slice`, `knots`, `svi`
(parameters, or null), `buttons` (as drawn: fitted + offsets), `k`, `delta`, `rmse_base`, `rmse_model` (the drawn smile),
`rmse_svi` (or null), `curve {strike, base, svi, model}` (400 points),
`nodes [{name, strike}]`,
`board {strike, is_call, bid, ask, mid, iv_bid, iv_ask, iv_mid, model_vol,
model_price, model_delta, volume, open_interest}`.

`/api/fit` also returns `bells_converged`; when it is false the stats line
says the bell fit did not converge.

Errors: `{"error": message}` with status 400 (bad input, ambiguous vols,
non-finite results), 403 (Host header other than 127.0.0.1 or localhost, which
blocks DNS-rebinding attacks from other websites), 404 (unknown slice or
route), 413 (body over 1 MB) or 500 (anything unexpected).

## Files

- `app/server.py`: HTTP server and API.
- `app/static/index.html`, `app.js`, `style.css`: the page (no external libraries).
- `tests/test_app.py`: starts the server on a free port and checks every
  endpoint against the library.
