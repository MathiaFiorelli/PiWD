from shiny import ui, App, render, reactive
import fastf1
import pandas as pd
from glob import glob
import fastf1.plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import numpy as np
import matplotlib as mpl
from matplotlib.collections import LineCollection
import shinywidgets as sw
import plotly.express as px


def unite_legends(axes):
    h, l = [], []
    for ax in axes:
        tmp = ax.get_legend_handles_labels()
        h.extend(tmp[0])
        l.extend(tmp[1])
    return h, l


def difference(main_x, main_y, aux_x, aux_y):
    error = []
    x_new = []
    y_new = []
    for x, y in zip(main_x, main_y):
        diff = [abs(x-x2) for x2 in aux_x]
        diff_min = min(diff)
        if diff_min < 2:
            x_new.append(aux_x[diff.index(diff_min)])
            y_new.append(aux_y[diff.index(diff_min)]-y)
            error.append(diff_min)
    return x_new, y_new, error


def distance(x1, y1, x2, y2):
    dist = np.sqrt((x1-x2)**2+(y1-y2)**2)
    return dist


def v_diff(x1, y1, v1, x2, y2, v2):
    if len(x1) < len(x2):
        x_main = x1
        y_main = y1
        v_main = v1
        x_aux = x2
        y_aux = y2
        v_aux = v2
        direction = 1
    else:
        x_main = x2
        y_main = y2
        v_main = v2
        x_aux = x1
        y_aux = y1
        v_aux = v1
        direction = -1
    v_diff_return = []
    for x, y, v in zip(x_main, y_main, v_main):
        d = [distance(x, y, x_i, y_i) for x_i, y_i in zip(x_aux, y_aux)]
        if min(d) < 1:
            idx = d.index(min(d))
            v_diff_return.append(v-v_aux[idx])
    return x_main, y_main, v_diff_return, direction


Events_df = pd.DataFrame(
    glob(r'C:\Users\mathi\OneDrive\Documents\Studia\SGH\PiWD\Projekt\App\CSV\*.csv'))
Events_df.rename(columns={0: 'Path'}, inplace=True)
Events_df['Event'] = Events_df['Path'].str[len(
    r'C:\Users\mathi\OneDrive\Documents\Studia\SGH\PiWD\Projekt\App/CSV/'):-4]
Events_df['DF'] = Events_df['Path'].apply(lambda x: pd.read_csv(x))
Events = sorted(Events_df['Event'].tolist())


sc_lap_times = ui.showcase_left_center(
    width='70%', width_full_screen='1fr', max_height='200px', max_height_full_screen='90%')

sc_lap_pos = ui.showcase_left_center(
    width='80%', width_full_screen='1fr', max_height='200px', max_height_full_screen='90%')

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize('event', 'Choose 2023 Event', Events),
        ui.input_slider("lap", "Lap", min=1, max=3, value=1, step=1),
        ui.input_selectize('driver', 'Choose Driver', [])
    ),
    ui.row(
        ui.output_ui("my_text", class_="display-3 text-center"),
        ui.layout_columns
        (
            *[ui.value_box('Lap', value=ui.output_text("lap_text"), theme='primary', showcase=ui.output_plot('lap_plot'), showcase_layout=sc_lap_times, full_screen=True),
              ui.value_box('Position', value=ui.output_text("position"), theme='primary',
                           showcase=ui.output_plot('pos_plot'), showcase_layout=sc_lap_pos, full_screen=True),
              ui.value_box('Compound', value=ui.output_text("compound"), theme='primary')],
            col_widths={'xs': (5, 5, 2)}
        )
    ),
    ui.row(
        ui.card(
            ui.card_header("Speed Comparison on Lap"),
            ui.output_plot("speed_plot"),
            full_screen=True
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Speed Contour"),
                ui.output_plot("contour_plot"),
                full_screen=True
            ),
            ui.card(
                ui.card_header("Speed Contour Comparison"),
                ui.output_plot("contour_diff"),
                full_screen=True
            ),
        ),
        ui.card(
            ui.card_header("Swarm-Violin Plot of Laptimes"),
            ui.output_plot("tyres_swarm"),
            full_screen=True,
            height='600px'
        ),
        ui.card(
            ui.card_header("Tyre Degradation Predictions"),
            ui.output_plot("tyres_line")
        ),
        ui.card(
            ui.card_header("Average vs Maximum Speed"),
            sw.output_widget("max_vs_avg"),
        )
    )
)


def server(input, output, session):

    @reactive.Effect()
    def _():
        event = input.event()
        df = Events_df.loc[Events_df['Event'] == event, 'DF'].iloc[0]
        laps = df['LapNumber'].max()
        drivers = sorted(df['Driver'].tolist())

        ui.update_slider('lap', label='Lap', min=1, max=laps, value=1)
        ui.update_select('driver', choices=drivers)

    @output
    @render.text
    def my_text():
        event = input.event()
        lap = input.lap()
        #driver = input.driver()
        text = f'{event}, Lap: {lap}'
        return text

    @reactive.Calc
    def df_event_driver():
        df_filtered = Events_df.loc[Events_df['Event'] == input.event(), 'DF'].iloc[0]
        df_filtered.drop([col for col in df_filtered.columns.tolist()
                         if 'Unnamed' in col], inplace=True, axis=1)

        if df_filtered['LapTime'].dtype == 'object':
            df_filtered['LapTime'] = pd.to_timedelta(df_filtered['LapTime'])

        if df_filtered['LapTime'].dtype == 'timedelta64[ns]':
            df_filtered['LapTime'] = df_filtered['LapTime'].dt.total_seconds()

        df_filtered['Color'] = df_filtered['Driver'].apply(
            lambda x: fastf1.plotting.driver_color(x))
        df_filtered = df_filtered.loc[df_filtered['Driver'] == input.driver()]
        return df_filtered

    @reactive.Calc
    def df_event():
        df_filtered = Events_df.loc[Events_df['Event'] == input.event(), 'DF'].iloc[0]
        df_filtered.drop([col for col in df_filtered.columns.tolist()
                         if 'Unnamed' in col], inplace=True, axis=1)

        if df_filtered['LapTime'].dtype == 'object':
            df_filtered['LapTime'] = pd.to_timedelta(df_filtered['LapTime'])

        if df_filtered['LapTime'].dtype == 'timedelta64[ns]':
            df_filtered['LapTime'] = df_filtered['LapTime'].dt.total_seconds()

        df_filtered['Color'] = df_filtered['Driver'].apply(
            lambda x: fastf1.plotting.driver_color(x))

        return df_filtered

    @render.text
    def lap_text():
        return lap_time_ret(df_event_driver(), input.lap())

    @render.text
    def position():
        return int(pos_ret(df_event_driver(), input.lap()))

    @render.text
    def compound():
        return comp_ret(df_event_driver(), input.lap())

    @output
    @render.plot
    def lap_plot():
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(figsize=(5, 2), dpi=150, layout='constrained')
        sns.set_style(rc={'axes.facecolor': 'darkgrey'})
        axes.set_title('Lap Times')
        axes.set_xlabel('Lap number [-]')
        axes.set_ylabel('Lap time [s]')
        axes.invert_yaxis()
        df = df_event_driver()
        lap = input.lap()
        sns_data = df.loc[(df['LapNumber'] <= lap)]
        sns.lineplot(data=sns_data, x='LapNumber', y='LapTime', hue='Compound', style='Stint', dashes=False, ax=axes, palette=fastf1.plotting.COMPOUND_COLORS,
                     hue_order=["SOFT", "MEDIUM", "HARD"])
        tmp = axes.get_legend_handles_labels()
        axes.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

        h = tmp[0][1:4]
        l = tmp[1][1:4]
        axes.legend(h, l, framealpha=1,
                    frameon=True, edgecolor='black', facecolor='white', loc=0, ncols=1)

        return fig

    @render.plot
    def pos_plot():
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(figsize=(10, 5), dpi=150, layout='constrained')
        sns.set_style(rc={'axes.facecolor': 'darkgrey'})
        axes.set_title('Postion change up to this lap')
        axes.set_xlabel('Lap number [-]')
        axes.set_ylabel('Position [-]')
        axes.invert_yaxis()
        df = df_event_driver()
        df = df.loc[(df['LapNumber'] <= input.lap())]
        sns.lineplot(data=df, x='LapNumber', y='Position')
        axes.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    @render.plot
    def speed_plot():
        # setting up the plot
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(figsize=(20, 8), dpi=150, layout='constrained')
        axes2 = axes.twinx()
        axes.set_title('Speed plot')
        axes.set_xlabel('Lap distance [%]')
        axes.set_ylabel('Speed [km/h]')
        axes2.set_ylabel('Difference (current - best) [km/h]')

        # Getting the inputs needed
        df = df_event()
        driver = input.driver()
        lap = input.lap()
        inputs = plot_inputs(df, driver, lap)
        d = inputs['d']
        v = inputs['v']
        tyre = inputs['tyre']
        t_life = inputs['t_life']
        c = inputs['c']
        d_diff = inputs['d_diff']
        v_diff_plot = inputs['v_diff_plot']
        d_best = inputs['d_best']
        v_best = inputs['v_best']
        driver_best = inputs['driver_best']
        lap_no_best = inputs['lap_no_best']
        tyre_best = inputs['tyre_best']
        t_life_best = inputs['t_life_best']
        c_best = inputs['c_best']

        # plotting
        axes.plot(
            d, v, label=f'{driver}, lap {int(lap)}, Compound: {tyre} ({int(t_life)} laps)', c=c)
        axes2.plot(d_diff, v_diff_plot, c='black', label='Difference in speed', ls='dotted')
        axes.plot(
            d_best, v_best, label=f'{driver_best}, lap {int(lap_no_best)}, Compound: {tyre_best} ({int(t_life_best)} laps)', c=c_best)

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

        handles, labels = unite_legends([axes, axes2])
        # ... HERE YOU SHOULD USE LAST axes INITIALIZED. IN THIS CASE IT'S axes_r
        axes2.legend(handles, labels, ncols=3, loc='upper center', framealpha=1,
                     frameon=True, edgecolor='black', bbox_to_anchor=(0.5, -0.3))
        return fig

    @render.plot
    def contour_plot():
        colormap = mpl.cm.inferno

        # Getting the inputs needed
        df = df_event()
        driver = input.driver()
        lap = input.lap()
        inputs = plot_inputs(df, driver, lap)
        x = inputs['x']
        y = inputs['y']
        v = inputs['v']

        points0 = np.array([x, y]).T.reshape(-1, 1, 2)
        segments0 = np.concatenate([points0[:-1], points0[1:]], axis=1)

        # We create a plot with title and adjust some setting to make it look good.
        fig, axes = plt.subplots(figsize=(10, 10), dpi=150, sharex=True, sharey=True)
        axes.set_title(f'{driver} - Speed')

        # Adjust margins and turn of axis
        axes.axis('off')

        # After this, we plot the data itself.
        # Create background track line
        axes.plot(x, y, color='black', linestyle='-', linewidth=10, zorder=0)

        # Create a continuous norm to map from data points to colors
        norm0 = plt.Normalize(min(v), max(v))
        lc0 = LineCollection(segments0, cmap=colormap, norm=norm0, linestyle='-', linewidth=5)

        # Set the values used for colormapping
        lc0.set_array(v)

        # Merge all line segments together
        axes.add_collection(lc0)
        cbar = plt.colorbar(lc0, ax=axes)
        cbar.set_label('Speed')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        return fig

    @render.plot
    def contour_diff():
        colormap1 = mpl.cm.seismic

        # Getting the inputs needed
        df = df_event()
        driver = input.driver()
        lap = input.lap()
        inputs = plot_inputs(df, driver, lap)
        v_diff_contour = inputs['v_diff_contour']
        x = inputs['x']
        y = inputs['y']
        x_main = inputs['x_main']
        y_main = inputs['y_main']
        driver_best = inputs['driver_best']

        points1 = np.array([x_main, y_main]).T.reshape(-1, 1, 2)
        segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)

        # We create a plot with title and adjust some setting to make it look good.
        fig, axes = plt.subplots(figsize=(10, 10), dpi=150, sharex=True, sharey=True)
        axes.set_title(f'{driver} current lap vs {driver_best} best lap\nSpeed Difference')
        # Adjust margins and turn of axis
        axes.axis('off')

        # After this, we plot the data itself.
        # Create background track line
        axes.plot(x, y, color='black', linestyle='-', linewidth=10, zorder=0)

        divnorm = mpl.colors.TwoSlopeNorm(vmin=float(
            min(v_diff_contour)), vcenter=0., vmax=float(max(v_diff_contour)))
        lc1 = LineCollection(segments1, cmap=colormap1, norm=divnorm, linestyle='-', linewidth=5)

        lc1.set_array(v_diff_contour)

        axes.add_collection(lc1)
        cbar = plt.colorbar(lc1, ax=axes)
        cbar.set_label('Speed')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        # Finally, we create a color bar as a legend.
        # cbaxes1 = fig.add_axes([0.95, 0.10, 0.02, 0.30])
        # legend1 = mpl.colorbar.ColorbarBase(
        #     norm=divnorm, cmap=colormap1, orientation="vertical", ax=cbaxes1)

        # legend1.set_label('Speed Difference')
        return fig

    @render.plot
    def tyres_swarm():
        sns.set_theme(style="darkgrid")
        sns.set_style(rc={'axes.facecolor': 'darkgrey'})
        fig, axes = plt.subplots(figsize=(10, 5), dpi=110)

        lap = input.lap()
        df = df_event()
        df_box = df.loc[(df['LapNumber'] <= lap) &
                        (df['TrackStatus'] == 1) & (df['PitOutTime'].isnull())]

        groups = df_box['Team'].tolist()
        colors = df_box['Color'].tolist()
        team_colors = {team: col for team, col in zip(groups, colors)}

        sns.violinplot(data=df_box,
                       x="Team",
                       y="LapTime",
                       inner=None,
                       density_norm='area',
                       palette=team_colors.values(), ax=axes
                       )

        sns.swarmplot(data=df_box,
                      x="Team",
                      y="LapTime",
                      hue="Compound",
                      palette=fastf1.plotting.COMPOUND_COLORS,
                      hue_order=["SOFT", "MEDIUM", "HARD"],
                      linewidth=0,
                      size=4, ax=axes
                      )

        axes.invert_yaxis()
        axes.legend(loc='lower right', framealpha=1, frameon=True,
                    facecolor='white', edgecolor='black')

        return fig

    @render.plot
    def tyres_line():
        sns.set_theme(style="darkgrid")
        sns.set_style(rc={'axes.facecolor': 'darkgrey'})
        fig, axes = plt.subplots(figsize=(20, 10), dpi=150, layout='constrained')

        lap = input.lap()
        df = df_event()
        df_box = df.loc[(df['LapNumber'] <= lap) &
                        (df['TrackStatus'] == 1) & (df['PitOutTime'].isnull())]

        sns.lineplot(data=df_box, x='TyreLife', y='LapTime',
                     style='Compound', hue='Compound', lw=3,
                     ax=axes,
                     palette=fastf1.plotting.COMPOUND_COLORS,
                     hue_order=["SOFT", "MEDIUM", "HARD"],
                     markers=True, dashes=False,  # estimator=None,
                     )

        axes.invert_yaxis()
        axes.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        axes.legend(framealpha=1, frameon=True, facecolor='white', edgecolor='black')
        return fig

    @sw.render_widget
    def max_vs_avg():
        df = df_event()
        df_box = df.loc[df['LapNumber'] == input.lap()]
        groups = df_box['Team'].tolist()
        colors = df_box['Color'].tolist()
        team_colors = {team: col for team, col in zip(groups, colors)}

        fig = px.scatter(
            df_box,
            x='Mean Speed',
            y='Max Speed',
            color='Team',
            hover_name='Driver',  # Display team name when hovering
            # Display driver and team in hover
            hover_data={'Driver': True, 'Team': True, 'Color': False},
            color_discrete_map=team_colors,  # Map team names to colors
            width=1200,
            height=800)
        fig.update_traces(marker_size=15)
        return fig


def lap_time_ret(df, lap):
    lap_time = df.loc[df['LapNumber'] == lap, 'LapTime'].iloc[0]
    return f'{lap_time} s'


def pos_ret(df, lap):
    pos = df.loc[df['LapNumber'] == lap, 'Position'].iloc[0]
    return pos


def comp_ret(df, lap):
    comp = df.loc[df['LapNumber'] == lap, 'Compound'].iloc[0]
    return comp


def lap_time_color(df, lap):
    lap_time = df.loc[df['LapNumber'] == lap, 'LapTime'].iloc[0]
    return lap_time


def plot_inputs(df, driver, lap):
    x = df.loc[(df['Driver'] == driver) & (df['LapNumber'] == lap), 'X'].iloc[0]
    x = x.strip('][').split(', ')
    x = [round(float(i)/10, 0) for i in x]

    y = df.loc[(df['Driver'] == driver) & (df['LapNumber'] == lap), 'Y'].iloc[0]
    y = y.strip('][').split(', ')
    y = [round(float(i)/10, 0) for i in y]

    v = df.loc[(df['Driver'] == driver) & (df['LapNumber'] == lap), 'Velocity'].iloc[0]
    v = v.strip('][').split(', ')
    v = [float(x) for x in v]

    d = df.loc[(df['Driver'] == driver) & (df['LapNumber'] == lap), 'Distance'].iloc[0]
    d = d.strip('][').split(', ')
    d = [float(x)*100 for x in d]

    tyre = df.loc[(df['Driver'] == driver) & (df['LapNumber'] == lap), 'Compound'].iloc[0]
    t_life = df.loc[(df['Driver'] == driver) & (df['LapNumber'] == lap), 'TyreLife'].iloc[0]
    c = df.loc[(df['Driver'] == driver) & (df['LapNumber'] == lap), 'Color'].iloc[0]

    lap_best = df['LapTime'].min()

    x_best = df.loc[df['LapTime'] == lap_best, 'X'].iloc[0]
    x_best = x_best.strip('][').split(', ')
    x_best = [round(float(i)/10, 0) for i in x_best]

    y_best = df.loc[df['LapTime'] == lap_best, 'Y'].iloc[0]
    y_best = y_best.strip('][').split(', ')
    y_best = [round(float(i)/10, 0) for i in y_best]

    v_best = df.loc[df['LapTime'] == lap_best, 'Velocity'].iloc[0]
    v_best = v_best.strip('][').split(', ')
    v_best = [float(x) for x in v_best]

    d_best = df.loc[df['LapTime'] == lap_best, 'Distance'].iloc[0]
    d_best = d_best.strip('][').split(', ')
    d_best = [float(x)*100 for x in d_best]

    tyre_best = df.loc[df['LapTime'] == lap_best, 'Compound'].iloc[0]
    t_life_best = df.loc[df['LapTime'] == lap_best, 'TyreLife'].iloc[0]
    driver_best = df.loc[df['LapTime'] == lap_best, 'Driver'].iloc[0]
    lap_no_best = df.loc[df['LapTime'] == lap_best, 'LapNumber'].iloc[0]
    c_best = df.loc[df['LapTime'] == lap_best, 'Color'].iloc[0]

    x_main, y_main, v_diff_contour, direction = v_diff(x, y, v, x_best, y_best, v_best)
    v_diff_contour = [v_x*direction for v_x in v_diff_contour]

    d_diff, v_diff_plot, error = difference(d_best, v_best, d, v)
    keys = ['x', 'y', 'v', 'd', 'tyre', 't_life', 'c', 'x_best', 'y_best', 'd_best', 'tyre_best',
            't_life_best', 'driver_best', 'lap_no_best', 'c_best', 'x_main', 'y_main', 'v_diff_contour', 'direction', 'd_diff', 'v_diff_plot', 'v_best']
    values = [x, y, v, d, tyre, t_life, c, x_best, y_best, d_best, tyre_best, t_life_best,
              driver_best, lap_no_best, c_best, x_main, y_main, v_diff_contour, direction, d_diff, v_diff_plot, v_best]
    dict_return = dict(zip(keys, values))
    print(c, c_best)
    return dict_return


app = App(app_ui, server)
