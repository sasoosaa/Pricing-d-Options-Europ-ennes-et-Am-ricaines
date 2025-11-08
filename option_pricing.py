import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
from scipy.stats import norm

# Initialisation
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY], suppress_callback_exceptions=True)
app.title = "Pricing d'Options - Modèles Stochastiques"
server = app.server


# ==================== FONCTIONS DE CALCUL ====================
# (Toutes les fonctions de calcul restent identiques)
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def binomial_price(S, K, T, r, sigma, N, option_type="call", option_style="european"):
    if T <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    
    # Initialisation des prix à l'échéance
    prices = np.zeros(N + 1)
    for i in range(N + 1):
        stock_price = S * (u ** i) * (d ** (N - i))
        prices[i] = max(stock_price - K, 0) if option_type == "call" else max(K - stock_price, 0)
    
    # Rétropolation
    for step in range(N - 1, -1, -1):
        for i in range(step + 1):
            stock_price = S * (u ** i) * (d ** (step - i))
            continuation_value = (p * prices[i + 1] + (1 - p) * prices[i]) * discount
            
            if option_style == "american":
                exercise_value = max(stock_price - K, 0) if option_type == "call" else max(K - stock_price, 0)
                prices[i] = max(continuation_value, exercise_value)
            else:
                prices[i] = continuation_value
    return prices[0]

def binomial_tree_data(S, K, T, r, sigma, N, option_type="call", option_style="european"):
    """Génère les données pour la visualisation de l'arbre binomial"""
    if T <= 0:
        return [], [], []
    
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    
    # Stock prices tree
    stock_tree = [[S]]
    for i in range(1, N + 1):
        level = []
        for j in range(i + 1):
            price = S * (u ** j) * (d ** (i - j))
            level.append(price)
        stock_tree.append(level)
    
    # Option prices tree
    option_tree = [[0] * (i + 1) for i in range(N + 1)]
    
    # Payoff at maturity
    for j in range(N + 1):
        stock_price = stock_tree[N][j]
        option_tree[N][j] = max(stock_price - K, 0) if option_type == "call" else max(K - stock_price, 0)
    
    # Backward induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = (p * option_tree[i + 1][j + 1] + (1 - p) * option_tree[i + 1][j]) * discount
            
            if option_style == "american":
                stock_price = stock_tree[i][j]
                exercise_value = max(stock_price - K, 0) if option_type == "call" else max(K - stock_price, 0)
                option_tree[i][j] = max(continuation_value, exercise_value)
            else:
                option_tree[i][j] = continuation_value
    
    return stock_tree, option_tree, p

def trinomial_price(S, K, T, r, sigma, N, option_type="call", option_style="european"):
    """Calcul du prix par modèle trinomial - IMPLÉMENTATION CORRECTE"""
    if T <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    
    dt = T / N
    
    # Paramètres standard du modèle trinomial
    u = np.exp(sigma * np.sqrt(3 * dt))
    d = 1 / u
    
    # Calcul des probabilités risk-neutres CORRECT
    nu = r - 0.5 * sigma**2
    dx = sigma * np.sqrt(3 * dt)
    
    # Probabilités risk-neutres standard
    q_up = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 + (nu * dt) / dx)
    q_down = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 - (nu * dt) / dx)
    q_mid = 1 - q_up - q_down
    
    # Vérification et ajustement des probabilités
    if q_up < 0 or q_down < 0 or q_mid < 0:
        # Ajustement pour garantir des probabilités positives
        q_up = max(0, q_up)
        q_down = max(0, q_down)
        total = q_up + q_down
        if total > 1:
            q_up /= total
            q_down /= total
        q_mid = 1 - q_up - q_down
    
    discount = np.exp(-r * dt)
    
    # Construction de l'arbre des prix de l'option
    # Nous avons 2N+1 nœuds à la maturité
    option_prices = [0] * (2 * N + 1)
    
    # Prix à l'échéance
    for i in range(2 * N + 1):
        # Calcul du prix du sous-jacent : i=0 (haut) à i=2N (bas)
        moves_from_center = N - i
        stock_price = S * (u ** moves_from_center)
        if option_type == "call":
            option_prices[i] = max(stock_price - K, 0)
        else:
            option_prices[i] = max(K - stock_price, 0)
    
    # Rétropolation
    for step in range(N - 1, -1, -1):
        new_prices = [0] * (2 * step + 1)
        for i in range(2 * step + 1):
            # Valeur de continuation
            continuation_value = (q_up * option_prices[i] + 
                                q_mid * option_prices[i + 1] + 
                                q_down * option_prices[i + 2]) * discount
            
            if option_style == "american":
                # Calcul du prix du sous-jacent pour l'exercice anticipé
                stock_price = S * (u ** (step - i))
                exercise_value = max(stock_price - K, 0) if option_type == "call" else max(K - stock_price, 0)
                new_prices[i] = max(continuation_value, exercise_value)
            else:
                new_prices[i] = continuation_value
        
        option_prices = new_prices
    
    return option_prices[0]

def trinomial_tree_data(S, K, T, r, sigma, N, option_type="call", option_style="european"):
    """Génère les données pour la visualisation de l'arbre trinomial - VERSION CORRECTE"""
    if T <= 0:
        return [], [], [], [], []
    
    dt = T / N
    
    # Mêmes paramètres que trinomial_price
    u = np.exp(sigma * np.sqrt(3 * dt))
    d = 1 / u
    
    nu = r - 0.5 * sigma**2
    dx = sigma * np.sqrt(3 * dt)
    
    q_up = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 + (nu * dt) / dx)
    q_down = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 - (nu * dt) / dx)
    q_mid = 1 - q_up - q_down
    
    # Ajustement des probabilités
    if q_up < 0 or q_down < 0 or q_mid < 0:
        q_up = max(0, q_up)
        q_down = max(0, q_down)
        total = q_up + q_down
        if total > 1:
            q_up /= total
            q_down /= total
        q_mid = 1 - q_up - q_down
    
    discount = np.exp(-r * dt)
    
    # Construction de l'arbre des prix du sous-jacent
    stock_tree = [[S]]
    for i in range(1, N + 1):
        level = []
        for j in range(2 * i + 1):
            moves_from_center = i - j
            price = S * (u ** moves_from_center)
            level.append(price)
        stock_tree.append(level)
    
    # Construction de l'arbre des prix de l'option
    option_tree = [[0] * (2 * i + 1) for i in range(N + 1)]
    
    # Payoff à la maturité
    for j in range(2 * N + 1):
        stock_price = stock_tree[N][j]
        option_tree[N][j] = max(stock_price - K, 0) if option_type == "call" else max(K - stock_price, 0)
    
    # Rétropolation
    for i in range(N - 1, -1, -1):
        for j in range(2 * i + 1):
            continuation_value = (q_up * option_tree[i + 1][j] + 
                                q_mid * option_tree[i + 1][j + 1] + 
                                q_down * option_tree[i + 1][j + 2]) * discount
            
            if option_style == "american":
                stock_price = stock_tree[i][j]
                exercise_value = max(stock_price - K, 0) if option_type == "call" else max(K - stock_price, 0)
                option_tree[i][j] = max(continuation_value, exercise_value)
            else:
                option_tree[i][j] = continuation_value
    
    return stock_tree, option_tree, q_up, q_mid, q_down

def calculate_bs_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    greeks = {}
    if option_type == "call":
        greeks['delta'] = norm.cdf(d1)
        greeks['gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        greeks['theta'] = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
        greeks['vega'] = S * np.sqrt(T) * norm.pdf(d1) / 100
        greeks['rho'] = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
    else:
        greeks['delta'] = norm.cdf(d1) - 1
        greeks['gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        greeks['theta'] = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r*T) * norm.cdf(-d2)) / 365
        greeks['vega'] = S * np.sqrt(T) * norm.pdf(d1) / 100
        greeks['rho'] = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
    return greeks

def plot_binomial_tree(stock_tree, option_tree, N_display=None):
    """Crée la visualisation de l'arbre binomial"""
    if not stock_tree or not option_tree:
        return go.Figure()
        
    if N_display is None:
        N_display = min(5, len(stock_tree) - 1)
    
    fig = go.Figure()
    
    # Coordinates for nodes
    x_coords = []
    y_coords = []
    stock_labels = []
    option_labels = []
    
    for i, level in enumerate(stock_tree[:N_display + 1]):
        for j, stock_price in enumerate(level):
            x = i
            y = j - i/2  # Center the tree
            x_coords.append(x)
            y_coords.append(y)
            stock_labels.append(f"{stock_price:.2f}")
            option_labels.append(f"{option_tree[i][j]:.2f}")
    
    # Add edges (lines between nodes)
    for i in range(min(N_display, len(stock_tree) - 1)):
        for j in range(len(stock_tree[i])):
            # Current node
            x1 = i
            y1 = j - i/2
            
            # Up node
            x2 = i + 1
            y2 = j + 1 - (i + 1)/2
            
            # Down node  
            y3 = j - (i + 1)/2
            
            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[y1, y2],
                mode='lines',
                line=dict(color='blue', width=1),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[y1, y3],
                mode='lines',
                line=dict(color='blue', width=1),
                showlegend=False
            ))
    
    # Create custom hover text
    hover_texts = []
    for i in range(len(stock_labels)):
        hover_text = f"Prix Actif: {stock_labels[i]}<br>Prix Option: {option_labels[i]}"
        hover_texts.append(hover_text)
    
    # Add stock price nodes
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers+text',
        marker=dict(size=15, color='lightblue'),
        text=stock_labels,
        textposition="middle center",
        name="Prix Actif",
        hovertext=hover_texts,
        hovertemplate="%{hovertext}<extra></extra>"
    ))
    
    # Add option price annotations
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        fig.add_annotation(
            x=x, y=y - 0.3,
            text=f"({option_labels[i]})",
            showarrow=False,
            font=dict(color="red", size=10)
        )
    
    fig.update_layout(
        title=f"Arbre Binomial (Affichage: {N_display} périodes)",
        xaxis_title="Périodes",
        yaxis_title="",
        showlegend=False,
        template="plotly_white",
        height=500
    )
    
    return fig

def plot_trinomial_tree(stock_tree, option_tree, N_display=None):
    """Crée la visualisation de l'arbre trinomial - VERSION CORRIGÉE"""
    if not stock_tree or not option_tree:
        return go.Figure()
        
    if N_display is None:
        N_display = min(4, len(stock_tree) - 1)
    
    fig = go.Figure()
    
    # Coordinates for nodes
    x_coords = []
    y_coords = []
    stock_labels = []
    option_labels = []
    
    for i, level in enumerate(stock_tree[:N_display + 1]):
        for j, stock_price in enumerate(level):
            x = i
            y = j - i  # Center the tree
            x_coords.append(x)
            y_coords.append(y)
            stock_labels.append(f"{stock_price:.2f}")
            option_labels.append(f"{option_tree[i][j]:.2f}")
    
    # CORRECTION : Connexions correctes entre les nœuds
    for i in range(min(N_display, len(stock_tree) - 1)):
        for j in range(len(stock_tree[i])):
            # Current node
            x1 = i
            y1 = j - i
            
            # Les trois nœuds enfants possibles
            for k in range(3):
                child_index = j + k
                if child_index < len(stock_tree[i + 1]):
                    x2 = i + 1
                    y2 = child_index - (i + 1)
                    
                    fig.add_trace(go.Scatter(
                        x=[x1, x2], y=[y1, y2],
                        mode='lines',
                        line=dict(color='orange', width=1),
                        showlegend=False
                    ))
    
    # Create custom hover text
    hover_texts = []
    for i in range(len(stock_labels)):
        hover_text = f"Prix Actif: {stock_labels[i]}<br>Prix Option: {option_labels[i]}"
        hover_texts.append(hover_text)
    
    # Add stock price nodes
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers+text',
        marker=dict(size=15, color='peachpuff'),
        text=stock_labels,
        textposition="middle center",
        name="Prix Actif",
        hovertext=hover_texts,
        hovertemplate="%{hovertext}<extra></extra>"
    ))
    
    # Add option price annotations
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        fig.add_annotation(
            x=x, y=y - 0.4,
            text=f"({option_labels[i]})",
            showarrow=False,
            font=dict(color="red", size=10)
        )
    
    fig.update_layout(
        title=f"Arbre Trinomial (Affichage: {N_display} périodes)",
        xaxis_title="Périodes",
        yaxis_title="",
        showlegend=False,
        template="plotly_white",
        height=500
    )
    
    return fig

# ==================== LAYOUTS AMÉLIORÉS ====================

home_layout = dbc.Container([
    dbc.Row([dbc.Col([
        html.H1("📊 Pricing d'Options Européennes & Américaines", className="display-4 fw-bold text-primary mb-4"),
        html.P("Comparez les modèles Black-Scholes, Binomial et Trinomial pour les options européennes et américaines.", className="lead text-dark fs-5"),
        html.Hr(className="my-4")
    ])]),
    
    # Section Modèles Implémentés
    dbc.Row([dbc.Col([html.H2("🧮 Modèles Implémentés", className="text-dark mb-4")])]),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H4("Black-Scholes", className="card-title text-primary"),
            html.P("Modèle analytique exact pour options européennes.", className="card-text"),
            html.Hr(),
            html.H6("Formule:", className="fw-bold"),
            html.P("C = S₀N(d₁) - Ke^(-rT)N(d₂)", className="small"),
            html.P("P = Ke^(-rT)N(-d₂) - S₀N(-d₁)", className="small"),
            html.H6("Avantages:", className="fw-bold mt-3"),
            html.Ul([
                html.Li("Solution analytique exacte"),
                html.Li("Calcul rapide"),
                html.Li("Grecques faciles à calculer")
            ], className="small"),
            html.H6("Limitations:", className="fw-bold mt-3"),
            html.Ul([
                html.Li("Options européennes uniquement"),
                html.Li("Hypothèses de marché restrictives"),
                html.Li("Pas d'exercice anticipé")
            ], className="small")
        ])], color="primary", outline=True, className="shadow-sm border-3 h-100"), md=4),
        
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H4("Modèle Binomial", className="card-title text-success"),
            html.P("Méthode numérique par arbre binomial.", className="card-text"),
            html.Hr(),
            html.H6("Principe:", className="fw-bold"),
            html.P("Arbre recombinant avec deux mouvements possibles à chaque étape", className="small"),
            html.H6("Paramètres:", className="fw-bold mt-3"),
            html.Ul([
                html.Li("u = e^(σ√Δt) (mouvement haussier)"),
                html.Li("d = 1/u (mouvement baissier)"),
                html.Li("p = (e^(rΔt) - d)/(u - d) (probabilité risque-neutre)")
            ], className="small"),
            html.H6("Avantages:", className="fw-bold mt-3"),
            html.Ul([
                html.Li("Options européennes et américaines"),
                html.Li("Intuitif et facile à implémenter"),
                html.Li("Converge vers Black-Scholes")
            ], className="small")
        ])], color="success", outline=True, className="shadow-sm border-3 h-100"), md=4),
        
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H4("Modèle Trinomial", className="card-title text-warning"),
            html.P("Extension à trois branches du binomial.", className="card-text"),
            html.Hr(),
            html.H6("Principe:", className="fw-bold"),
            html.P("Arbre avec trois mouvements possibles: haussier, stable, baissier", className="small"),
            html.H6("Paramètres:", className="fw-bold mt-3"),
            html.Ul([
                html.Li("u = e^(σ√(3Δt))"),
                html.Li("d = 1/u"),
                html.Li("q₊, q₀, q₋ (probabilités des trois états)")
            ], className="small"),
            html.H6("Avantages:", className="fw-bold mt-3"),
            html.Ul([
                html.Li("Convergence plus rapide que le binomial"),
                html.Li("Plus flexible"),
                html.Li("Options européennes et américaines")
            ], className="small")
        ])], color="warning", outline=True, className="shadow-sm border-3 h-100"), md=4),
    ], className="gy-3 mb-5"),
    
    # Section Théorie et Contexte
    dbc.Row([dbc.Col([html.H2("📚 Théorie et Contexte", className="text-dark mb-4")])]),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H4("Options Européennes vs Américaines", className="text-info"),
            html.Hr(),
            html.H6("Options Européennes:", className="fw-bold"),
            html.Ul([
                html.Li("Exercice uniquement à la date d'échéance"),
                html.Li("Prix généralement inférieur aux options américaines"),
                html.Li("Modélisables par Black-Scholes")
            ]),
            html.H6("Options Américaines:", className="fw-bold mt-3"),
            html.Ul([
                html.Li("Exercice possible à tout moment avant échéance"),
                html.Li("Prime supplémentaire pour la flexibilité"),
                html.Li("Nécessitent des méthodes numériques (arbres)")
            ])
        ])], className="h-100"), md=6),
        
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H4("Concepts Clés", className="text-info"),
            html.Hr(),
            html.H6("Probabilité Risque-Neutre:", className="fw-bold"),
            html.P("Monde fictif où les investisseurs sont indifférents au risque", className="small"),
            html.H6("Grecques:", className="fw-bold mt-3"),
            html.Ul([
                html.Li("Delta: Sensibilité au prix du sous-jacent"),
                html.Li("Gamma: Sensibilité du delta"),
                html.Li("Theta: Sensibilité au temps"),
                html.Li("Vega: Sensibilité à la volatilité"),
                html.Li("Rho: Sensibilité aux taux d'intérêt")
            ], className="small")
        ])], className="h-100"), md=6),
    ], className="gy-3 mb-5"),
    
    # Section Paramètres et Formules
    dbc.Row([dbc.Col([html.H2("🔢 Paramètres et Formules", className="text-dark mb-4")])]),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H4("Paramètres Communs", className="text-primary"),
            html.Hr(),
            html.Ul([
                html.Li(html.Strong("S₀: "), "Prix actuel du sous-jacent"),
                html.Li(html.Strong("K: "), "Prix d'exercice (strike)"),
                html.Li(html.Strong("T: "), "Temps jusqu'à l'échéance (années)"),
                html.Li(html.Strong("r: "), "Taux d'intérêt sans risque"),
                html.Li(html.Strong("σ: "), "Volatilité du sous-jacent")
            ])
        ])], className="h-100"), md=4),
        
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H4("Formules Black-Scholes", className="text-success"),
            html.Hr(),
            html.P(html.Strong("d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)"), className="small"),
            html.P(html.Strong("d₂ = d₁ - σ√T"), className="small mb-3"),
            html.P("Call = S₀N(d₁) - Ke^(-rT)N(d₂)", className="small"),
            html.P("Put = Ke^(-rT)N(-d₂) - S₀N(-d₁)", className="small")
        ])], className="h-100"), md=4),
        
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H4("Convergence", className="text-warning"),
            html.Hr(),
            html.P("Les modèles binomial et trinomial convergent vers Black-Scholes quand:"),
            html.Ul([
                html.Li("Nombre de périodes → ∞"),
                html.Li("Δt = T/N → 0"),
                html.Li("Paramètres correctement calibrés")
            ], className="small")
        ])], className="h-100"), md=4),
    ], className="gy-3 mb-5"),
    
    # Section Mode d'emploi
    dbc.Row([dbc.Col([html.H2("📖 Mode d'emploi", className="text-dark mb-3")])]),
    dbc.Row([
        dbc.Col([
            dbc.Card([dbc.CardBody([
                html.H4("Comment utiliser cette application", className="text-primary"),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H5("1. Sélection du modèle", className="text-success"),
                            html.P("Choisissez le modèle dans le menu latéral:", className="mb-2"),
                            html.Ul([
                                html.Li("Black-Scholes pour les options européennes"),
                                html.Li("Binomial pour une approche intuitive"),
                                html.Li("Trinomial pour une convergence rapide")
                            ])
                        ], className="mb-4"),
                        
                        html.Div([
                            html.H5("2. Paramétrage", className="text-success"),
                            html.P("Entrez les paramètres du contrat:", className="mb-2"),
                            html.Ul([
                                html.Li("Prix actuel (S₀) et strike (K)"),
                                html.Li("Maturité (T) en années"),
                                html.Li("Taux d'intérêt (r) et volatilité (σ)")
                            ])
                        ], className="mb-4")
                    ], md=6),
                    
                    dbc.Col([
                        html.Div([
                            html.H5("3. Options avancées", className="text-success"),
                            html.P("Pour les modèles d'arbres:", className="mb-2"),
                            html.Ul([
                                html.Li("Choisissez le style (européen/américain)"),
                                html.Li("Ajustez le nombre de périodes"),
                                html.Li("Visualisez l'arbre de prix")
                            ])
                        ], className="mb-4"),
                        
                        html.Div([
                            html.H5("4. Analyse des résultats", className="text-success"),
                            html.P("Interprétez les résultats:", className="mb-2"),
                            html.Ul([
                                html.Li("Prix de l'option"),
                                html.Li("Grecques (Black-Scholes)"),
                                html.Li("Erreur de convergence"),
                                html.Li("Visualisation graphique")
                            ])
                        ])
                    ], md=6)
                ])
            ])])
        ])
    ], className="mb-5"),
    
    # Section Références
    dbc.Row([dbc.Col([html.H2("📚 Références", className="text-dark mb-4")])]),
    dbc.Row([
        dbc.Col([
            dbc.Card([dbc.CardBody([
                html.H5("Bibliographie", className="text-primary"),
                html.Hr(),
                html.Ul([
                    html.Li("Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities"),
                    html.Li("Cox, J.C., Ross, S.A. & Rubinstein, M. (1979). Option Pricing: A Simplified Approach"),
                    html.Li("Hull, J.C. (2022). Options, Futures and Other Derivatives"),
                    html.Li("Shreve, S.E. (2004). Stochastic Calculus for Finance")
                ], className="small")
            ])])
        ])
    ], className="mb-5")
], fluid=True, className="py-4")

# Layout Black-Scholes amélioré avec cases claires
blackscholes_layout = dbc.Container([
    html.H2("📈 Modèle Black-Scholes", className="text-primary my-4"),
    dbc.Card([dbc.CardBody([
        html.H4("Paramètres", className="card-title text-primary"),
        dbc.Row([
            dbc.Col([dbc.Label("Prix actuel (S₀)"), dbc.Input(id="bs-S0", type="number", value=100, min=0.1, step=1)], md=3),
            dbc.Col([dbc.Label("Strike (K)"), dbc.Input(id="bs-K", type="number", value=100, min=0.1, step=1)], md=3),
            dbc.Col([dbc.Label("Maturité (T)"), dbc.Input(id="bs-T", type="number", value=1.0, min=0.01, step=0.1)], md=3),
            dbc.Col([dbc.Label("Type"), dcc.Dropdown(id="bs-type", options=[{"label": "Call", "value": "call"}, {"label": "Put", "value": "put"}], value="call")], md=3)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([dbc.Label("Taux (r)"), dbc.Input(id="bs-r", type="number", value=0.05, step=0.01)], md=4),
            dbc.Col([dbc.Label("Volatilité (σ)"), dbc.Input(id="bs-sigma", type="number", value=0.2, step=0.01)], md=4),
            dbc.Col([dbc.Button("🔄 Calculer", id="bs-calc-btn", color="primary", className="mt-4 w-100")], md=4)
        ])
    ])], className="mb-4"),
    
    dbc.Row([
        # Graphique principal
        dbc.Col([dcc.Graph(id="bs-graph")], md=8),
        
        # NOUVELLE SECTION : Prix et Grecques dans des cases claires
        dbc.Col([
            # Case pour le prix de l'option
            dbc.Card([
                dbc.CardHeader(html.H4("💰 Prix de l'Option", className="text-success mb-0")),
                dbc.CardBody([
                    html.Div(id="bs-price-display", className="text-center", 
                            style={"fontSize": "2rem", "fontWeight": "bold", "color": "#28a745"})
                ])
            ], className="mb-3 shadow-sm border-success", style={"backgroundColor": "#f8fff9"}),
            
            # Case pour les grecques
            dbc.Card([
                dbc.CardHeader(html.H4("📊 Grecques", className="text-info mb-0")),
                dbc.CardBody([
                    html.Div(id="bs-greeks-display", className="fs-6")
                ])
            ], className="shadow-sm border-info", style={"backgroundColor": "#f8f9ff"})
        ], md=4)
    ]),
    
    # Graphiques de sensibilité
    dbc.Row([
        dbc.Col([dcc.Graph(id="bs-sensitivity-s")], md=4),
        dbc.Col([dcc.Graph(id="bs-sensitivity-t")], md=4),
        dbc.Col([dcc.Graph(id="bs-sensitivity-sigma")], md=4)
    ], className="mt-4")
], fluid=True)

# Layout Binomial amélioré avec cases claires
binomial_layout = dbc.Container([
    html.H2("🌳 Modèle Binomial", className="text-success my-4"),
    dbc.Card([dbc.CardBody([
        html.H4("Paramètres", className="card-title text-success"),
        dbc.Row([
            dbc.Col([dbc.Label("Prix actuel (S₀)"), dbc.Input(id="bin-S0", type="number", value=100, min=0.1, step=1)], md=3),
            dbc.Col([dbc.Label("Strike (K)"), dbc.Input(id="bin-K", type="number", value=100, min=0.1, step=1)], md=3),
            dbc.Col([dbc.Label("Maturité (T)"), dbc.Input(id="bin-T", type="number", value=1.0, min=0.01, step=0.1)], md=3),
            dbc.Col([dbc.Label("Type"), dcc.Dropdown(id="bin-type", options=[{"label": "Call", "value": "call"}, {"label": "Put", "value": "put"}], value="call")], md=3)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([dbc.Label("Taux (r)"), dbc.Input(id="bin-r", type="number", value=0.05, step=0.01)], md=3),
            dbc.Col([dbc.Label("Volatilité (σ)"), dbc.Input(id="bin-sigma", type="number", value=0.2, step=0.01)], md=3),
            dbc.Col([dbc.Label("Périodes (N)"), dbc.Input(id="bin-N", type="number", value=10, min=1, step=1)], md=3),
            dbc.Col([dbc.Label("Style d'option"), dcc.Dropdown(
                id="bin-style", 
                options=[
                    {"label": "Européenne", "value": "european"}, 
                    {"label": "Américaine", "value": "american"}
                ], 
                value="european"
            )], md=3)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([dbc.Label("Périodes à afficher"), 
                    dcc.Slider(id="bin-display-slider", min=1, max=10, step=1, value=5,
                              marks={i: str(i) for i in range(1, 11)})], md=6),
            dbc.Col([dbc.Button("🔄 Calculer", id="bin-calc-btn", color="success", className="mt-4 w-100")], md=6)
        ])
    ])], className="mb-4"),
    
    dbc.Row([
        dbc.Col([dcc.Graph(id="bin-convergence-graph")], md=6),
        dbc.Col([dcc.Graph(id="bin-tree-graph")], md=6)
    ]),
    
    # NOUVELLE SECTION : Résultats dans des cases claires
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("💰 Prix de l'Option", className="text-success mb-0")),
                dbc.CardBody([
                    html.Div(id="bin-price-display", className="text-center",
                            style={"fontSize": "2rem", "fontWeight": "bold", "color": "#28a745"})
                ])
            ], className="mb-3 shadow-sm border-success", style={"backgroundColor": "#f8fff9"})
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("📊 Détails du Calcul", className="text-info mb-0")),
                dbc.CardBody([
                    html.Div(id="bin-results", className="fs-6")
                ])
            ], className="shadow-sm border-info", style={"backgroundColor": "#f8f9ff"})
        ], md=8)
    ])
], fluid=True)

# Layout Trinomial amélioré avec cases claires
trinomial_layout = dbc.Container([
    html.H2("🔺 Modèle Trinomial", className="text-warning my-4"),
    dbc.Card([dbc.CardBody([
        html.H4("Paramètres", className="card-title text-warning"),
        dbc.Row([
            dbc.Col([dbc.Label("Prix actuel (S₀)"), dbc.Input(id="tri-S0", type="number", value=100, min=0.1, step=1)], md=3),
            dbc.Col([dbc.Label("Strike (K)"), dbc.Input(id="tri-K", type="number", value=100, min=0.1, step=1)], md=3),
            dbc.Col([dbc.Label("Maturité (T)"), dbc.Input(id="tri-T", type="number", value=1.0, min=0.01, step=0.1)], md=3),
            dbc.Col([dbc.Label("Type"), dcc.Dropdown(id="tri-type", options=[{"label": "Call", "value": "call"}, {"label": "Put", "value": "put"}], value="call")], md=3)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([dbc.Label("Taux (r)"), dbc.Input(id="tri-r", type="number", value=0.05, step=0.01)], md=3),
            dbc.Col([dbc.Label("Volatilité (σ)"), dbc.Input(id="tri-sigma", type="number", value=0.2, step=0.01)], md=3),
            dbc.Col([dbc.Label("Périodes (N)"), dbc.Input(id="tri-N", type="number", value=50, min=1, step=1)], md=3),
            dbc.Col([dbc.Label("Style d'option"), dcc.Dropdown(
                id="tri-style", 
                options=[
                    {"label": "Européenne", "value": "european"}, 
                    {"label": "Américaine", "value": "american"}
                ], 
                value="european"
            )], md=3)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([dbc.Label("Périodes à afficher"), 
                    dcc.Slider(id="tri-display-slider", min=1, max=10, step=1, value=4,
                              marks={i: str(i) for i in range(1, 11)})], md=6),
            dbc.Col([dbc.Button("🔄 Calculer", id="tri-calc-btn", color="warning", className="mt-4 w-100")], md=6)
        ])
    ])], className="mb-4"),
    
    dbc.Row([
        dbc.Col([dcc.Graph(id="tri-convergence-graph")], md=6),
        dbc.Col([dcc.Graph(id="tri-tree-graph")], md=6)
    ]),
    
    # NOUVELLE SECTION : Résultats dans des cases claires
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("💰 Prix de l'Option", className="text-warning mb-0")),
                dbc.CardBody([
                    html.Div(id="tri-price-display", className="text-center",
                            style={"fontSize": "2rem", "fontWeight": "bold", "color": "#ffc107"})
                ])
            ], className="mb-3 shadow-sm border-warning", style={"backgroundColor": "#fffbf0"})
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("📊 Détails du Calcul", className="text-info mb-0")),
                dbc.CardBody([
                    html.Div(id="tri-results", className="fs-6")
                ])
            ], className="shadow-sm border-info", style={"backgroundColor": "#f8f9ff"})
        ], md=8)
    ])
], fluid=True)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div([
        html.H2("📈 Options Pricer", className="mb-4 text-white"),
        dcc.Link("🏠 Accueil", href="/", className="nav-link", id="link-home"),
        dcc.Link("📊 Black-Scholes", href="/blackscholes", className="nav-link", id="link-bs"),
        dcc.Link("🌳 Binomial", href="/binomial", className="nav-link", id="link-bin"),
        dcc.Link("🔺 Trinomial", href="/trinomial", className="nav-link", id="link-tri"),
    ], className="sidebar"),
    html.Div(id="page-content", className="main-content")
])

# ==================== CALLBACKS AMÉLIORÉS ====================

@callback([Output("link-home", "className"), Output("link-bs", "className"), 
           Output("link-bin", "className"), Output("link-tri", "className")], Input("url", "pathname"))
def highlight_active_link(pathname):
    classes = ["nav-link"] * 4
    if pathname == "/": classes[0] += " active"
    elif pathname == "/blackscholes": classes[1] += " active"
    elif pathname == "/binomial": classes[2] += " active"
    elif pathname == "/trinomial": classes[3] += " active"
    return classes

@callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/blackscholes": return blackscholes_layout
    elif pathname == "/binomial": return binomial_layout
    elif pathname == "/trinomial": return trinomial_layout
    else: return home_layout

@callback([Output("bs-graph", "figure"), 
           Output("bs-sensitivity-s", "figure"), 
           Output("bs-sensitivity-t", "figure"), 
           Output("bs-sensitivity-sigma", "figure"),
           Output("bs-price-display", "children"),
           Output("bs-greeks-display", "children")],
          Input("bs-calc-btn", "n_clicks"),
          [State("bs-S0", "value"), State("bs-K", "value"), State("bs-T", "value"),
           State("bs-r", "value"), State("bs-sigma", "value"), State("bs-type", "value")])
def update_blackscholes(n_clicks, S0, K, T, r, sigma, option_type):
    if n_clicks is None:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), "0.00€", "Cliquez sur Calculer"
    try:
        S0, K, T, r, sigma = float(S0), float(K), float(T), float(r), float(sigma)
        price = black_scholes_price(S0, K, T, r, sigma, option_type)
        greeks = calculate_bs_greeks(S0, K, T, r, sigma, option_type)
        
        # Graphique principal : Prix en fonction du strike
        strikes = np.linspace(S0 * 0.5, S0 * 1.5, 50)
        prices = [black_scholes_price(S0, k, T, r, sigma, option_type) for k in strikes]
        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(x=strikes, y=prices, mode='lines', line=dict(color='blue', width=3), name='Prix'))
        fig_main.add_vline(x=K, line_dash="dash", line_color="red", annotation_text=f"K={K}")
        fig_main.update_layout(title=f"Option {option_type.capitalize()} - Black-Scholes", 
                         xaxis_title="Strike", yaxis_title="Prix", template="plotly_white")
        
        # Graphique de sensibilité au prix du sous-jacent
        S_range = np.linspace(S0 * 0.5, S0 * 1.5, 50)
        prices_S = [black_scholes_price(s, K, T, r, sigma, option_type) for s in S_range]
        fig_sensitivity_S = go.Figure()
        fig_sensitivity_S.add_trace(go.Scatter(x=S_range, y=prices_S, mode='lines', line=dict(color='green', width=2), name='Prix'))
        fig_sensitivity_S.add_vline(x=S0, line_dash="dash", line_color="red", annotation_text=f"S₀={S0}")
        fig_sensitivity_S.update_layout(title="Sensibilité au prix du sous-jacent",
                                  xaxis_title="Prix du sous-jacent (S₀)", yaxis_title="Prix de l'option",
                                  template="plotly_white", height=300)
        
        # Graphique de sensibilité au temps
        T_range = np.linspace(0.01, T * 2, 50)
        prices_T = [black_scholes_price(S0, K, t, r, sigma, option_type) for t in T_range]
        fig_sensitivity_T = go.Figure()
        fig_sensitivity_T.add_trace(go.Scatter(x=T_range, y=prices_T, mode='lines', line=dict(color='orange', width=2), name='Prix'))
        fig_sensitivity_T.add_vline(x=T, line_dash="dash", line_color="red", annotation_text=f"T={T}")
        fig_sensitivity_T.update_layout(title="Sensibilité au temps",
                                  xaxis_title="Temps jusqu'à maturité (années)", yaxis_title="Prix de l'option",
                                  template="plotly_white", height=300)
        
        # Graphique de sensibilité à la volatilité
        sigma_range = np.linspace(0.01, 0.5, 50)
        prices_sigma = [black_scholes_price(S0, K, T, r, s, option_type) for s in sigma_range]
        fig_sensitivity_sigma = go.Figure()
        fig_sensitivity_sigma.add_trace(go.Scatter(x=sigma_range, y=prices_sigma, mode='lines', line=dict(color='purple', width=2), name='Prix'))
        fig_sensitivity_sigma.add_vline(x=sigma, line_dash="dash", line_color="red", annotation_text=f"σ={sigma}")
        fig_sensitivity_sigma.update_layout(title="Sensibilité à la volatilité",
                                      xaxis_title="Volatilité (σ)", yaxis_title="Prix de l'option",
                                      template="plotly_white", height=300)
        
        # Affichage du prix dans une case dédiée
        price_display = f"{price:.4f}€"
        
        # Affichage des grecques formaté
        greeks_display = html.Div([
            dbc.Row([
                dbc.Col(html.Strong("Delta:"), width=4),
                dbc.Col(f"{greeks['delta']:.4f}", className="text-end")
            ], className="mb-2 border-bottom"),
            dbc.Row([
                dbc.Col(html.Strong("Gamma:"), width=4),
                dbc.Col(f"{greeks['gamma']:.4f}", className="text-end")
            ], className="mb-2 border-bottom"),
            dbc.Row([
                dbc.Col(html.Strong("Theta:"), width=4),
                dbc.Col(f"{greeks['theta']:.4f}", className="text-end")
            ], className="mb-2 border-bottom"),
            dbc.Row([
                dbc.Col(html.Strong("Vega:"), width=4),
                dbc.Col(f"{greeks['vega']:.4f}", className="text-end")
            ], className="mb-2 border-bottom"),
            dbc.Row([
                dbc.Col(html.Strong("Rho:"), width=4),
                dbc.Col(f"{greeks['rho']:.4f}", className="text-end")
            ], className="mb-2")
        ])
        
        return fig_main, fig_sensitivity_S, fig_sensitivity_T, fig_sensitivity_sigma, price_display, greeks_display
        
    except Exception as e:
        error_msg = f"Erreur: {str(e)}"
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), "Erreur", error_msg

@callback([Output("bin-convergence-graph", "figure"), 
           Output("bin-tree-graph", "figure"), 
           Output("bin-price-display", "children"),
           Output("bin-results", "children")],
          Input("bin-calc-btn", "n_clicks"),
          [State("bin-S0", "value"), State("bin-K", "value"), State("bin-T", "value"),
           State("bin-r", "value"), State("bin-sigma", "value"), State("bin-N", "value"), 
           State("bin-type", "value"), State("bin-style", "value"), State("bin-display-slider", "value")])
def update_binomial(n_clicks, S0, K, T, r, sigma, N, option_type, option_style, display_periods):
    if n_clicks is None:
        return go.Figure(), go.Figure(), "0.00€", "Cliquez sur Calculer"
    try:
        S0, K, T, r, sigma, N = float(S0), float(K), float(T), float(r), float(sigma), int(N)
        price = binomial_price(S0, K, T, r, sigma, N, option_type, option_style)
        
        # Générer les données de l'arbre
        stock_tree, option_tree, p = binomial_tree_data(S0, K, T, r, sigma, N, option_type, option_style)
        
        # Pour les options européennes, on compare avec Black-Scholes
        bs_price = None
        if option_style == "european":
            bs_price = black_scholes_price(S0, K, T, r, sigma, option_type)
        
        # COURBE DE CONVERGENCE AVEC DÉTECTION D'OSCILLATION AMÉLIORÉE
        max_step = min(200, N * 2)
        # Augmenter le nombre de points pour mieux voir les oscillations
        periods = list(range(1, max_step + 1, 1))  # Pas de 1 pour maximum de points
        bin_prices = []
        
        print(f"Calcul de {len(periods)} points de convergence...")
        
        for n in periods:
            try:
                bin_price = binomial_price(S0, K, T, r, sigma, n, option_type, option_style)
                bin_prices.append(bin_price)
            except Exception as e:
                bin_prices.append(np.nan)
                print(f"Erreur pour n={n}: {e}")
        
        convergence_fig = go.Figure()
        
        # Courbe principale avec style qui montre mieux les oscillations
        convergence_fig.add_trace(go.Scatter(
            x=periods, 
            y=bin_prices, 
            mode='lines+markers',
            name=f'Binomial {option_style}', 
            line=dict(color='green', width=2),
            marker=dict(size=3, opacity=0.6),
            hovertemplate='Périodes: %{x}<br>Prix: %{y:.6f}€<extra></extra>'
        ))
        
        # DÉTECTION SENSIBLE DES OSCILLATIONS
        bin_highs = []
        bin_lows = []
        
        if bin_prices and not all(np.isnan(bin_prices)):
            # Calculer la moyenne mobile pour détecter la tendance
            window_size = max(3, len(periods) // 50)  # Fenêtre adaptative
            smoothed_prices = []
            
            for i in range(len(bin_prices)):
                start = max(0, i - window_size)
                end = min(len(bin_prices), i + window_size + 1)
                window = [p for p in bin_prices[start:end] if not np.isnan(p)]
                if window:
                    smoothed_prices.append(np.mean(window))
                else:
                    smoothed_prices.append(np.nan)
            
            # Détection des extremums locaux par rapport à la courbe lissée
            for i in range(1, len(bin_prices)-1):
                if (np.isnan(bin_prices[i]) or np.isnan(smoothed_prices[i]) or
                    np.isnan(bin_prices[i-1]) or np.isnan(bin_prices[i+1])):
                    continue
                
                current_price = bin_prices[i]
                smoothed = smoothed_prices[i]
                
                # Détection des pics (au-dessus de la courbe lissée)
                if (current_price > smoothed and 
                    current_price > bin_prices[i-1] and 
                    current_price > bin_prices[i+1]):
                    bin_highs.append((periods[i], current_price))
                
                # Détection des creux (en-dessous de la courbe lissée)
                elif (current_price < smoothed and 
                      current_price < bin_prices[i-1] and 
                      current_price < bin_prices[i+1]):
                    bin_lows.append((periods[i], current_price))
        
        # Points hauts - marqueurs visibles
        if bin_highs:
            highs_x, highs_y = zip(*bin_highs)
            convergence_fig.add_trace(go.Scatter(
                x=highs_x, 
                y=highs_y, 
                mode='markers',
                name='Pics',
                marker=dict(size=8, color='red', symbol='triangle-up', line=dict(width=2, color='darkred')),
                hovertemplate='<b>PIC</b><br>Périodes: %{x}<br>Prix: %{y:.6f}€<extra></extra>'
            ))
        
        # Points bas - marqueurs visibles
        if bin_lows:
            lows_x, lows_y = zip(*bin_lows)
            convergence_fig.add_trace(go.Scatter(
                x=lows_x, 
                y=lows_y, 
                mode='markers',
                name='Creux',
                marker=dict(size=8, color='blue', symbol='triangle-down', line=dict(width=2, color='darkblue')),
                hovertemplate='<b>CREUX</b><br>Périodes: %{x}<br>Prix: %{y:.6f}€<extra></extra>'
            ))
        
        # Ligne de tendance lissée
        if bin_prices and not all(np.isnan(bin_prices)):
            window_size = max(5, len(periods) // 30)
            trend_line = []
            for i in range(len(bin_prices)):
                start = max(0, i - window_size)
                end = min(len(bin_prices), i + window_size + 1)
                window = [p for p in bin_prices[start:end] if not np.isnan(p)]
                if window:
                    trend_line.append(np.mean(window))
                else:
                    trend_line.append(np.nan)
            
            convergence_fig.add_trace(go.Scatter(
                x=periods, 
                y=trend_line, 
                mode='lines', 
                name='Tendance',
                line=dict(color='black', width=2, dash='dash'),
                hovertemplate='Tendance: %{y:.6f}€<extra></extra>'
            ))
        
        if bs_price is not None:
            # Ligne Black-Scholes
            convergence_fig.add_hline(
                y=bs_price, 
                line_dash="dash", 
                line_color="purple", 
                annotation_text=f"Black-Scholes = {bs_price:.6f}",
                annotation_position="bottom right"
            )
            
            # Calcul et affichage de l'erreur
            errors = [abs(bp - bs_price) if not np.isnan(bp) else np.nan for bp in bin_prices]
            
            convergence_fig.add_trace(go.Scatter(
                x=periods, 
                y=errors, 
                mode='lines', 
                name='Erreur absolue', 
                line=dict(color='orange', width=2, dash='dot'),
                yaxis='y2',
                hovertemplate='Erreur: %{y:.6f}€<extra></extra>'
            ))
            
            convergence_fig.update_layout(
                yaxis2=dict(
                    title="Erreur absolue (€)",
                    overlaying='y',
                    side='right',
                    showgrid=False,
                    color='orange'
                )
            )
        
        # Métriques d'oscillation
        if bin_prices and not all(np.isnan(bin_prices)):
            valid_prices = [p for p in bin_prices if not np.isnan(p)]
            if valid_prices:
                price_range = max(valid_prices) - min(valid_prices)
                oscillations_count = len(bin_highs) + len(bin_lows)
                
                if bs_price and bs_price != 0:
                    oscillation_pct = (price_range / bs_price) * 100
                else:
                    oscillation_pct = 0
                
                convergence_fig.add_annotation(
                    x=0.02, y=0.98,
                    xref="paper", yref="paper",
                    text=f"Oscillations détectées: {oscillations_count}<br>Amplitude: {price_range:.6f}€<br>Soit {oscillation_pct:.2f}%",
                    showarrow=False,
                    bgcolor="lightyellow",
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(size=10)
                )
        
        convergence_fig.update_layout(
            title=f"Convergence du modèle Binomial - Oscillations Visibles ({option_style.capitalize()})",
            xaxis_title="Nombre de périodes (N)",
            yaxis_title="Prix de l'option (€)",
            template="plotly_white",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=600,
            hovermode='x unified'
        )
        
        # Graphique de l'arbre
        tree_fig = plot_binomial_tree(stock_tree, option_tree, display_periods)
        
        # Affichage du prix
        price_display = f"{price:.6f}€"
        
        # Résultats avec informations détaillées sur les oscillations
        if bs_price is not None and bin_prices and not all(np.isnan(bin_prices)):
            valid_prices = [p for p in bin_prices if not np.isnan(p)]
            current_error = abs(price - bs_price)
            error_pct = (current_error / bs_price) * 100 if bs_price != 0 else 0
            price_range = max(valid_prices) - min(valid_prices)
            oscillation_pct = (price_range / bs_price) * 100 if bs_price != 0 else 0
            oscillations_count = len(bin_highs) + len(bin_lows)
            
            comparison_text = f"""Prix Black-Scholes: {bs_price:.6f}€
Erreur absolue: {current_error:.6f}€
Erreur relative: {error_pct:.4f}%
Amplitude totale: {price_range:.6f}€
Oscillation: {oscillation_pct:.2f}%
Nombre d'oscillations: {oscillations_count}"""
        else:
            comparison_text = "Comparaison BS: Non disponible pour options américaines"
            price_range = 0
            oscillation_pct = 0
            oscillations_count = 0
        
        results = f"""Prix actuel: {S0:.2f}€
Style: {option_style.capitalize()}
Type: {option_type.capitalize()}

{comparison_text}

Paramètres:
• Strike: {K:.2f}€
• Périodes: {N}
• Probabilité p: {p:.4f}
• Volatilité: {sigma*100:.2f}%
• Taux: {r*100:.2f}%"""
        
        return convergence_fig, tree_fig, price_display, results
    except Exception as e:
        return go.Figure(), go.Figure(), "Erreur", f"Erreur: {str(e)}"

@callback([Output("tri-convergence-graph", "figure"), 
           Output("tri-tree-graph", "figure"), 
           Output("tri-price-display", "children"),
           Output("tri-results", "children")],
          Input("tri-calc-btn", "n_clicks"),
          [State("tri-S0", "value"), State("tri-K", "value"), State("tri-T", "value"), State("tri-r", "value"),
           State("tri-sigma", "value"), State("tri-N", "value"), 
           State("tri-type", "value"), State("tri-style", "value"), State("tri-display-slider", "value")])
def update_trinomial(n_clicks, S0, K, T, r, sigma, N, option_type, option_style, display_periods):
    if n_clicks is None:
        return go.Figure(), go.Figure(), "0.00€", "Cliquez sur Calculer"
    try:
        S0, K, T, r, sigma, N = float(S0), float(K), float(T), float(r), float(sigma), int(N)
        
        price = trinomial_price(S0, K, T, r, sigma, N, option_type, option_style)
        
        # Générer les données de l'arbre
        stock_tree, option_tree, q_up, q_mid, q_down = trinomial_tree_data(S0, K, T, r, sigma, N, option_type, option_style)
        
        # Pour les options européennes, on compare avec Black-Scholes
        bs_price = None
        if option_style == "european":
            bs_price = black_scholes_price(S0, K, T, r, sigma, option_type)
        
        # COURBE DE CONVERGENCE AVEC DÉTECTION D'OSCILLATION AMÉLIORÉE
        max_step = min(200, N * 2)
        periods = list(range(1, max_step + 1, 1))  # Pas de 1 pour maximum de points
        tri_prices = []
        
        print(f"Calcul de {len(periods)} points de convergence trinomial...")
        
        for n in periods:
            try:
                tri_price = trinomial_price(S0, K, T, r, sigma, n, option_type, option_style)
                tri_prices.append(tri_price)
            except Exception as e:
                tri_prices.append(np.nan)
                print(f"Erreur pour n={n}: {e}")
        
        convergence_fig = go.Figure()
        
        # Courbe principale
        convergence_fig.add_trace(go.Scatter(
            x=periods, 
            y=tri_prices, 
            mode='lines+markers',
            name=f'Trinomial {option_style}', 
            line=dict(color='orange', width=2),
            marker=dict(size=3, opacity=0.6),
            hovertemplate='Périodes: %{x}<br>Prix: %{y:.6f}€<extra></extra>'
        ))
        
        # DÉTECTION SENSIBLE DES OSCILLATIONS
        tri_highs = []
        tri_lows = []
        
        if tri_prices and not all(np.isnan(tri_prices)):
            # Calculer la moyenne mobile pour détecter la tendance
            window_size = max(3, len(periods) // 50)
            smoothed_prices = []
            
            for i in range(len(tri_prices)):
                start = max(0, i - window_size)
                end = min(len(tri_prices), i + window_size + 1)
                window = [p for p in tri_prices[start:end] if not np.isnan(p)]
                if window:
                    smoothed_prices.append(np.mean(window))
                else:
                    smoothed_prices.append(np.nan)
            
            # Détection des extremums locaux
            for i in range(1, len(tri_prices)-1):
                if (np.isnan(tri_prices[i]) or np.isnan(smoothed_prices[i]) or
                    np.isnan(tri_prices[i-1]) or np.isnan(tri_prices[i+1])):
                    continue
                
                current_price = tri_prices[i]
                smoothed = smoothed_prices[i]
                
                # Détection des pics
                if (current_price > smoothed and 
                    current_price > tri_prices[i-1] and 
                    current_price > tri_prices[i+1]):
                    tri_highs.append((periods[i], current_price))
                
                # Détection des creux
                elif (current_price < smoothed and 
                      current_price < tri_prices[i-1] and 
                      current_price < tri_prices[i+1]):
                    tri_lows.append((periods[i], current_price))
        
        # Points hauts
        if tri_highs:
            highs_x, highs_y = zip(*tri_highs)
            convergence_fig.add_trace(go.Scatter(
                x=highs_x, 
                y=highs_y, 
                mode='markers',
                name='Pics',
                marker=dict(size=8, color='red', symbol='triangle-up', line=dict(width=2, color='darkred')),
                hovertemplate='<b>PIC</b><br>Périodes: %{x}<br>Prix: %{y:.6f}€<extra></extra>'
            ))
        
        # Points bas
        if tri_lows:
            lows_x, lows_y = zip(*tri_lows)
            convergence_fig.add_trace(go.Scatter(
                x=lows_x, 
                y=lows_y, 
                mode='markers',
                name='Creux',
                marker=dict(size=8, color='blue', symbol='triangle-down', line=dict(width=2, color='darkblue')),
                hovertemplate='<b>CREUX</b><br>Périodes: %{x}<br>Prix: %{y:.6f}€<extra></extra>'
            ))
        
        # Ligne de tendance lissée
        if tri_prices and not all(np.isnan(tri_prices)):
            window_size = max(5, len(periods) // 30)
            trend_line = []
            for i in range(len(tri_prices)):
                start = max(0, i - window_size)
                end = min(len(tri_prices), i + window_size + 1)
                window = [p for p in tri_prices[start:end] if not np.isnan(p)]
                if window:
                    trend_line.append(np.mean(window))
                else:
                    trend_line.append(np.nan)
            
            convergence_fig.add_trace(go.Scatter(
                x=periods, 
                y=trend_line, 
                mode='lines', 
                name='Tendance',
                line=dict(color='black', width=2, dash='dash'),
                hovertemplate='Tendance: %{y:.6f}€<extra></extra>'
            ))
        
        if bs_price is not None:
            # Ligne Black-Scholes
            convergence_fig.add_hline(
                y=bs_price, 
                line_dash="dash", 
                line_color="purple", 
                annotation_text=f"Black-Scholes = {bs_price:.6f}",
                annotation_position="bottom right"
            )
            
            # Calcul et affichage de l'erreur
            errors = [abs(tp - bs_price) if not np.isnan(tp) else np.nan for tp in tri_prices]
            
            convergence_fig.add_trace(go.Scatter(
                x=periods, 
                y=errors, 
                mode='lines', 
                name='Erreur absolue', 
                line=dict(color='orange', width=2, dash='dot'),
                yaxis='y2',
                hovertemplate='Erreur: %{y:.6f}€<extra></extra>'
            ))
            
            convergence_fig.update_layout(
                yaxis2=dict(
                    title="Erreur absolue (€)",
                    overlaying='y',
                    side='right',
                    showgrid=False,
                    color='orange'
                )
            )
        
        # Métriques d'oscillation
        if tri_prices and not all(np.isnan(tri_prices)):
            valid_prices = [p for p in tri_prices if not np.isnan(p)]
            if valid_prices:
                price_range = max(valid_prices) - min(valid_prices)
                oscillations_count = len(tri_highs) + len(tri_lows)
                
                if bs_price and bs_price != 0:
                    oscillation_pct = (price_range / bs_price) * 100
                else:
                    oscillation_pct = 0
                
                convergence_fig.add_annotation(
                    x=0.02, y=0.98,
                    xref="paper", yref="paper",
                    text=f"Oscillations détectées: {oscillations_count}<br>Amplitude: {price_range:.6f}€<br>Soit {oscillation_pct:.2f}%",
                    showarrow=False,
                    bgcolor="lightyellow",
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(size=10)
                )
        
        convergence_fig.update_layout(
            title=f"Convergence du modèle Trinomial - Oscillations Visibles ({option_style.capitalize()})",
            xaxis_title="Nombre de périodes (N)",
            yaxis_title="Prix de l'option (€)",
            template="plotly_white",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=600,
            hovermode='x unified'
        )
        
        # Graphique de l'arbre
        tree_fig = plot_trinomial_tree(stock_tree, option_tree, min(display_periods, N))
        
        # Affichage du prix
        price_display = f"{price:.6f}€"
        
        # Résultats avec informations détaillées
        if bs_price is not None and tri_prices and not all(np.isnan(tri_prices)):
            valid_prices = [p for p in tri_prices if not np.isnan(p)]
            current_error = abs(price - bs_price)
            error_pct = (current_error / bs_price) * 100 if bs_price != 0 else 0
            price_range = max(valid_prices) - min(valid_prices)
            oscillation_pct = (price_range / bs_price) * 100 if bs_price != 0 else 0
            oscillations_count = len(tri_highs) + len(tri_lows)
            
            comparison_text = f"""Prix Black-Scholes: {bs_price:.6f}€
Erreur absolue: {current_error:.6f}€
Erreur relative: {error_pct:.4f}%
Amplitude totale: {price_range:.6f}€
Oscillation: {oscillation_pct:.2f}%
Nombre d'oscillations: {oscillations_count}"""
        else:
            comparison_text = "Comparaison BS: Non disponible pour options américaines"
            price_range = 0
            oscillation_pct = 0
            oscillations_count = 0
        
        results = f"""Prix actuel: {S0:.2f}€
Style: {option_style.capitalize()}
Type: {option_type.capitalize()}

{comparison_text}

Paramètres:
• Strike: {K:.2f}€
• Périodes: {N}
• Probabilités:
  - q₊ (haussier): {q_up:.4f}
  - q₀ (stable): {q_mid:.4f}  
  - q₋ (baissier): {q_down:.4f}
• Volatilité: {sigma*100:.2f}%
• Taux: {r*100:.2f}%"""
        
        return convergence_fig, tree_fig, price_display, results
    except Exception as e:
        return go.Figure(), go.Figure(), "Erreur", f"Erreur: {str(e)}"

# ==================== STYLE CSS AMÉLIORÉ ====================

app.index_string = '''
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
body{margin:0;font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);}
.sidebar{position:fixed;left:0;top:0;bottom:0;width:250px;background:linear-gradient(180deg,#2c3e50 0%,#34495e 100%);padding:30px 20px;box-shadow:2px 0 10px rgba(0,0,0,0.1);z-index:1000;}
.sidebar h2{color:white;font-size:24px;margin-bottom:30px;text-align:center;font-weight:bold;}
.nav-link{display:block;color:#ecf0f1;text-decoration:none;padding:15px 20px;margin:10px 0;border-radius:8px;transition:all 0.3s;font-size:16px;font-weight:500;}
.nav-link:hover{background:rgba(255,255,255,0.1);transform:translateX(5px);color:white;}
.nav-link.active{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;box-shadow:0 4px 15px rgba(102,126,234,0.4);}
.main-content{margin-left:250px;padding:40px;min-height:100vh;background:white;}
.card{border-radius:15px;box-shadow:0 4px 6px rgba(0,0,0,0.1);transition:transform 0.3s,box-shadow 0.3s;}
.card:hover{transform:translateY(-5px);box-shadow:0 8px 15px rgba(0,0,0,0.2);}
.btn{border-radius:8px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;transition:all 0.3s;}
.btn:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,0.2);}
.form-control,.form-select{border-radius:8px;border:2px solid #e0e0e0;transition:border-color 0.3s;}
.form-control:focus,.form-select:focus{border-color:#667eea;box-shadow:0 0 0 0.2rem rgba(102,126,234,0.25);}
h1,h2,h3,h4{font-weight:700;}
.display-4{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}

/* Styles pour les cases de prix et grecques */
.price-card {background: linear-gradient(135deg, #f8fff9 0%, #e8f5e8 100%);}
.greeks-card {background: linear-gradient(135deg, #f8f9ff 0%, #e8eaff 100%);}
.greek-item {border-bottom: 1px solid #e0e0e0; padding: 8px 0;}
.greek-item:last-child {border-bottom: none;}

@media (max-width:768px){.sidebar{width:100%;position:relative;height:auto;}.main-content{margin-left:0;}}
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=False, port=8050)
    

    
