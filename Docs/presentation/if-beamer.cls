\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{if-beamer}[2026/01/01 v1.0 Custom IF-Beamer Class for IIT Mandi MTP Presentations]

% Pass all unknown options to beamer
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}
\ProcessOptions\relax

% Load base beamer class with widescreen aspect ratio by default
\LoadClass[aspectratio=169, 10pt]{beamer}

% Core packages
\RequirePackage{tikz}
\RequirePackage{xcolor}
\RequirePackage{amsmath,amssymb}
\RequirePackage{booktabs}
\RequirePackage{textpos}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc, fit, backgrounds, shadows}

% IIT Mandi Color Palette
\definecolor{iitmaroon}{RGB}{140, 29, 34}       % IIT Mandi Maroon #8C1D22
\definecolor{iittwilight}{RGB}{35, 47, 62}      % Deep Charcoal #232F3E
\definecolor{accentorange}{RGB}{193, 68, 14}    % Terracotta / Warm Accent #C1440E
\definecolor{techblue}{RGB}{20, 82, 140}        % Secondary Engineering Blue #14528C
\definecolor{cardbg}{RGB}{248, 249, 250}        % Crisp clean background
\definecolor{darktext}{RGB}{30, 30, 30}
\definecolor{lightgray}{RGB}{230, 233, 238}

% Beamer Theme Customization
\useinnertheme{rectangles}
\useoutertheme{infolines}

% Color assignments
\setbeamercolor{structure}{fg=iitmaroon}
\setbeamercolor{normal text}{fg=darktext}
\setbeamercolor{title}{fg=white, bg=iitmaroon}
\setbeamercolor{subtitle}{fg=white!90}
\setbeamercolor{frametitle}{fg=white, bg=iitmaroon}
\setbeamercolor{framesubtitle}{fg=white!85}

\setbeamercolor{section in head/foot}{fg=white, bg=iittwilight}
\setbeamercolor{author in head/foot}{fg=white, bg=iitmaroon}
\setbeamercolor{title in head/foot}{fg=white, bg=iitmaroon!90!black}
\setbeamercolor{date in head/foot}{fg=white, bg=iittwilight}

% Blocks
\setbeamercolor{block title}{fg=white, bg=iitmaroon}
\setbeamercolor{block body}{fg=darktext, bg=cardbg}
\setbeamercolor{block title alerted}{fg=white, bg=accentorange}
\setbeamercolor{block body alerted}{fg=darktext, bg=accentorange!8}
\setbeamercolor{block title example}{fg=white, bg=techblue}
\setbeamercolor{block body example}{fg=darktext, bg=techblue!8}

% Itemize and enumerate styling
\setbeamercolor{item}{fg=iitmaroon}
\setbeamercolor{subitem}{fg=accentorange}
\setbeamertemplate{itemize items}[square]
\setbeamertemplate{enumerate items}[default]

% Clean navigation symbols
\setbeamertemplate{navigation symbols}{}

% Logo macro compatibility
\newcommand{\logoleft}[1]{\def\@logoleft{#1}}
\let\@logoleft\@empty

% Custom Title Page
\setbeamertemplate{title page}{%
  \begin{minipage}[t][\textheight][t]{\textwidth}
    \vspace{0.2cm}
    \begin{beamercolorbox}[wd=\textwidth, sep=10pt, rounded=true, shadow=false]{title}
      {\centering
        {\usebeamerfont{title}\Large\bfseries\inserttitle\par}%
        \ifx\insertsubtitle\@empty\else%
          \vspace{0.4em}%
          {\usebeamerfont{subtitle}\small\bfseries\insertsubtitle\par}%
        \fi%
      }
    \end{beamercolorbox}
    \vspace{0.5cm}
    \begin{columns}[T]
      \begin{column}{0.65\textwidth}
        \textbf{\insertauthor}\\[0.4em]
        \scriptsize\insertinstitute\\[0.3em]
        \textit{\insertdate}
      \end{column}
      \begin{column}{0.32\textwidth}
        \raggedleft
        \ifx\@logoleft\@empty
          \begin{tikzpicture}
            \draw[fill=iitmaroon!10, draw=iitmaroon, thick, rounded corners=4pt] (0,0) rectangle (3.2, 1.2);
            \node[align=center, font=\scriptsize\bfseries, text=iitmaroon] at (1.6, 0.6) {IIT MANDI\\MTP CS399P};
          \end{tikzpicture}
        \else
          \@logoleft
        \fi
      \end{column}
    \end{columns}
  \end{minipage}
}

% Frame title styling with clean line
\setbeamertemplate{frametitle}{%
  \begin{beamercolorbox}[wd=\paperwidth, ht=2.8ex, dp=1.2ex, leftskip=0.8cm, rightskip=0.8cm]{frametitle}
    \textbf{\insertframetitle}
    \ifx\insertframesubtitle\@empty\else
      \hfill{\small\mdseries\insertframesubtitle}
    \fi
  \end{beamercolorbox}%
}

% Footline with slide numbering
\setbeamertemplate{footline}{%
  \leavevmode%
  \hbox{%
    \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,center]{author in head/foot}%
      \usebeamerfont{author in head/foot}\insertshortauthor
    \end{beamercolorbox}%
    \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,center]{title in head/foot}%
      \usebeamerfont{title in head/foot}\insertshorttitle
    \end{beamercolorbox}%
    \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,right]{date in head/foot}%
      \usebeamerfont{date in head/foot}CS399P Mid-Evaluation\hspace*{2em}
      \insertframenumber{} / \inserttotalframenumber\hspace*{2ex}
    \end{beamercolorbox}}%
  \vskip0pt%
}

\endinput
