package jnyqide;

// Code: Chris Yealy, 5/2/06
// Edited: Derek D'Souza 5/3/07
// Edited: Roger Dannenberg 23Jul07

// eq editor needs to do same with workspace

/* Saving and Restoring:
 *  the Save button saves one envelope locally, to Nyquist, and saves the
 *   updated workspace
 *  the Load button loads all envelopes from Nyquist for editing. If no
 *   envelopes are loaded, suggests (load "workspace")
 *  selecting an envelope from the list saves the current envelope
 *   (locally, to Nyquist, and to the workspace) and switches the editor
 *   to the selected envelope
 *  delete button removes the envelope from the workspace after a confirm
 */

/* Representation:
If the user puts a non-zero point at time zero or at maxT, that indicates
a PWLV or PWEV function. 
If there is no point at time zero, one is implied at 0,0 for PWL and
0,1 for PWEV.
If there is no point at time maxT, one is implied at maxT,0 for PWL and
maxT,1 for PWEV.
Implied points are not stored in the params data, but they are generated
when the EnvData is converted to a function call string.
For PWL, the params data points are t1, v1, t2, v2, ..., tn-1, vn-1,
0,0 is implied, and maxT stores tn.
For PWLV, the params data points are t0, v0, t1, v1, t2, v2, ..., tn, vn, 
where either T0=0.0 or maxT is tn.
For PWEV, the params are the same as for either PWLV except it is not
necessary for either end point to be present in the data.

When the maxTime increases, a new point should be inserted at (oldMaxT, 0)
if this is a PWL, or at (oldMaxT, y) and (newMaxT, y), where y is the old 
final value.
When the maxTime decreases, do not allow it to take effect if there is 
an explicit point beyond the newMaxTime unless the function is constant
from newMaxTime to oldMaxTime, in which case, just change the time of the
last point from oldMaxTime to newMaxTime.

*/


/*
 Graphics organization:

 myParent -- the MainFrame
 EnvelopeFrame -- this editor, a JInternalFrame
 mainPanel -- a JPanel
 topPanel -- a JPanel
 envNamePanel -- a JPanel, border "Current Envelope"
 currEnvNameLabel -- a JLabel("Name:")
 currEnvName -- a JTextField("ENVELOPE", 20)
 saveEnv -- a JButtton("Save")
 envTypeLabel - a JTextField("Type", ...)
 envTypes - a JComboBox
 envListPanel -- a JPanel, border "Saved Envelopes List"
 envName -- a JComboBox
 loadEnv -- a JButton("load")
 deleteEnv -- a JButton("delete")
 envPointsPanel - a JPanel, border Envelope Points
 envTimesLabel - a JLabel("Time:")
 envTimes - JComboBox
 envAmplitudesLabel - a JLabel("Amplitudes:")
 envAmplitudes -  JTextField
 addPoint - a JButton("Add Point")
 deletePoint - a JButton("Delete Point")
 updatePoint - a JButton("Update Point")
 paramPanel -- a JPanel
 rangePanel - a JPanel, border "Range Parameters"
 maxTL - a JLabel("Stop Time")
 maxT - a JTextField
 minAL - a JLabel("Min. Amplitude")
 maxAL - a JLabel("Max. Amplitude")
 minA - a JTextField
 maxA - a JTextField
 update - JButton("Update Range")
 comparePanel -- a JPanel, border "Show More Envelopes"
 compare1 - a JComboBox
 compare2 - a JComboBox
 compare3 - a JComboBox
 compare4 - a JComboBox
 gridPanel -- a JPanel, border "Grid"
 updateGrid - a JButton("Update Grid")
 snap - a JButton("Snap Points to Grid")
 toggleGrid - a Checkbox("Display Grid")
 timeResL - a JLabel("Time Res.")
 timeRes - a JTextField
 ampResL - a JLabel("Time Ampl.")
 ampRes - a JTextField
 envEditPanel -- a JPanel, border "Graphic Envelope Editor"
 envEditButtonPanel - a JPanel
 undo - a JButton("Undo")
 redo - a JButton("Redo")
 clear - a JButton("Clear")
 dispCoord - a JButton("Coordinates")
 output - a JButton("Output Envelope")
 copy - a JButton("Copy to Clipboard")
 canvas - a PiecewiseCanvas (a JPanel)

 to drag point:
 on mouse pressed, set selection to index of point
 on mouse drag, delete selection and insert new point

 if endpoints are specified, e.g. if times[0] is 0 or last value in times
 matches the ending time, then use PWLV or PWEV version
 Be careful not to drop first/last point unless the time matches the
 begin/end time.


 */

import java.awt.*;
import java.awt.event.*;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import javax.swing.event.*;
import java.awt.image.BufferedImage;
import java.util.Vector;
import java.util.Hashtable;
import java.util.StringTokenizer;
import java.util.Enumeration;
import java.io.File;
import java.lang.Math.*;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;

import jnyqide.*;



/**
 * Internal Frame for envolope editing. Special version of Internal frame is
 * used to avoid being moved outside the desktop panel.
 * 
 * @author original_author Revised by Zeyu Jin
 * 
 */
public class EnvelopeFrame extends JNonHideableInternalFrame implements
        ActionListener {

    double EPSILON = 0.00000001; // small number -- allow for rounding error
    int LEFT_BORDER = 3; // inset everything by 3 pixels
    int TOP_BORDER = 3;
    int BOTTOM_BORDER = 5;
    int RIGHT_BORDER = 5;
    MainFrame myParent;

    // JComponents for envelope window
    private JComboBox envTypes;
    private JPanel mainPanel;
    private JTextPane codeInput;
    // private JPanel canvasPanel;
    private PiecewiseCanvas canvas;

    private JTextField envAmplitudes;
    // private JTextField minT;
    private JTextField maxT;
    private JTextField minA;
    private JTextField maxA;
    private JTextField timeRes;
    private JTextField ampRes;
    private JTextField currEnvName;
    private JComboBox envName;
    private JComboBox compare1;
    private JComboBox compare2;
    private JComboBox compare3;
    private JComboBox compare4;
    private JComboBox envTimes;
    private JButton saveEnv;
    private JToggleButton dispCoord;
    private JButton loadEnv;
    private JButton deleteEnv;
    private JButton addPoint;
    private JButton deletePoint;
    private JButton updatePoint;
    private JButton updateRange;
    private JButton snap;
    private JButton updateGrid;
    private JButton undo;
    private JButton redo;
    private JButton clear;
    private JButton output;
    private JButton copy;
    private JToggleButton toggleGrid;
    private String currentEnvName;
    private EnvelopeFrame thisFrame;
    public String[] types = { "Piecewise Linear", "Piecewise Exponential" };
    public int PWL_TYPE = 0;
    public int PWE_TYPE = 1;
    // when saving envelopes, we copy current envelope to the collection,
    // but this makes the combobox think that an envelope was selected,
    // and we get a request to save the same envelope were saving. To
    // avoid this, the "saving" flag is used to disable the check.
    private boolean saving = false;
    // When an item is deleted, the combo box selects a new element.
    // Unlike the above situation, we want the display to update.
    // However, we need to disable the prompt.
    private boolean deleting = false;
    // Envelopes with negative amplitudes may have undefined behavior.
    // We want to warn the user if they try to use a negative amplitude.
    // However, displaying this every time the window is changed is annoying.
    // Use a bool so that the warning is only displayed once per session.
    private boolean promptNegativeAmp = true;

    // hashtable for storing envelopes
    private Hashtable<String, String> envColl;

    static double initTime = 0.0;
    static double finalTime = 1.0;
    static boolean displayCoord = false;
    static boolean valueType = false;
    static DecimalFormat form = new DecimalFormat("#.###");
    private boolean modified; // tells when any aspect of envelope has changed

    // envelope is modified by: entering a point, deleting a point, changing
    // the end-time (update), or clearing
    // modified is reset when the envelope is loaded or saved

    // Constructor
    public EnvelopeFrame(final MainFrame parent, JTextPane inputArea) {
        super();
        setPreferredSize(new Dimension(520, 520));
        setMinimumSize(new Dimension(520, 520));
        thisFrame = this;
        modified = false;
        constructFrame(parent, inputArea);
        constructData();

        System.out.println("EnvelopeFrame constructor 1");
        initGlobalComponents();

        // components for parameter panel
        ActionListener stateChange = new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                canvas.history.save(canvas.data);
            }
        };

	// since form is static, there's probably a better way or place for this:
	// fix format to use "." as decimal point for XLISP compatibility
	DecimalFormatSymbols symbols = form.getDecimalFormatSymbols();
	symbols.setDecimalSeparator('.');
	symbols.setGroupingSeparator(','); // should never be used
	form.setDecimalFormatSymbols(symbols);

        constructComponents();
        setSize(500, 515);
        pack();

        setResizable(true);
        setVisible(true);
        setClosable(true);
        setMaximizable(true);
        setIconifiable(true);
        loadEnvelopes();
        repaint();
    }

    private void constructFrame(final MainFrame parent, JTextPane inputArea) {
        setTitle("Piecewise Envelope Generator");
        myParent = parent;
        codeInput = inputArea;
        // canvasPanel = new JPanel();
        // canvasPanel.setPreferredSize(new Dimension(575, 256));
        mainPanel = (JPanel) getContentPane();
        setDefaultCloseOperation(JInternalFrame.DO_NOTHING_ON_CLOSE);
        addInternalFrameListener(new InternalFrameListener() {
            public void internalFrameClosing(InternalFrameEvent e) {
                // System.out.println("FrameClosing");
                int r = JOptionPane.OK_OPTION;
                if (thisFrame.modified) {
                    r = JOptionPane.showConfirmDialog(thisFrame,
                            "Really close without saving?", "Alert",
                            JOptionPane.OK_CANCEL_OPTION);
                }
                if (r == JOptionPane.OK_OPTION) {
                    thisFrame.dispose();
                }
            }

            public void internalFrameOpened(InternalFrameEvent e) {
            }

            public void internalFrameClosed(InternalFrameEvent e) {
                parent.disconnectEnv();
                // System.out.println("FrameClosed");
            }

            public void internalFrameIconified(InternalFrameEvent e) {
            }

            public void internalFrameDeiconified(InternalFrameEvent e) {
            }

            public void internalFrameActivated(InternalFrameEvent e) {
            }

            public void internalFrameDeactivated(InternalFrameEvent e) {
            }
        });
    }


    private void constructData() {
        envColl = new Hashtable<String, String>();
    }


    private boolean okToChangeMaxT(Vector<Double> params, double newMaxT, 
                                   double oldMaxT, double defaultValue) {
        // it is ok to reduct MaxT if the last explicit time point in 
        // the envelope is less than newMaxT or the envelope is
        // constant after newMaxT
        if (params.size() == 0) return true; // no points
        // ok if newMaxT is beyond last point
        int i = params.size() - 2;
        if (newMaxT >= params.get(i)) return true;
        // check if function is constant beyond newMaxT
        // first, figure out final value
        double finalValue = defaultValue;
        if (within(oldMaxT, params.get(i), EPSILON)) {
            finalValue = params.get(i + 1);
        }
        // now see if final value holds before newMaxT
        while (i >= 0) {
            if (!within(params.get(i + 1), finalValue, EPSILON)) return false;
            if (params.get(i) <= newMaxT + EPSILON) return true;
            i -= 2;
        }
        // first point is not before newMaxT, so check implied point
        return within(defaultValue, finalValue, EPSILON);
    }


    private void initGlobalComponents() {
        //
        currEnvName = new JTextField("envelope", 10);
        currentEnvName = "envelope";

        //
        saveEnv = new JButton("Save");
        saveEnv.addActionListener(this);
        saveEnv.setActionCommand("saveEnvelopes");

        //
        envTypes = new JComboBox(types);
        envTypes.addActionListener(this);
        envTypes.setActionCommand("envTypeChanged");

        // components for envelope list panel
        envName = new JComboBox();
        envName.setEditable(false);
        envName.setPreferredSize(currEnvName.getPreferredSize());
        envName.addActionListener(this);
        envName.setActionCommand("envNameSelected");

        //
        loadEnv = new JButton("Load");
        loadEnv.addActionListener(this);
        loadEnv.setActionCommand("loadEnvelope");

        //
        deleteEnv = new JButton("Delete");
        deleteEnv.addActionListener(this);
        deleteEnv.setActionCommand("deleteEnvelope");

        //
        envTimes = new JComboBox();
        envTimes.setMaximumSize(new Dimension(68, 27));
        envTimes.setPreferredSize(new Dimension(64, 27));
        envTimes.setEditable(true);
        envTimes.addActionListener(this);
        envTimes.setActionCommand("envTimeChange");
        /*
        envTimes.addItemListener(new ItemListener() {
            public void itemStateChanged(ItemEvent e) {
                System.out.println("itemStateChanged " + e);
            }
        });
        */
        //
        envAmplitudes = new JTextField(6);

        //
        addPoint = new JButton("Add Point");
        addPoint.addActionListener(this);
        addPoint.setActionCommand("addPoint");

        //
        deletePoint = new JButton("Delete Point");
        deletePoint.addActionListener(this);
        deletePoint.setActionCommand("deletePoint");

        //
        updatePoint = new JButton("Update Point");
        updatePoint.addActionListener(this);
        updatePoint.setActionCommand("updatePoint");

        //
        maxT = new JTextField("1.0", 5);
        minA = new JTextField("0.0", 5);
        maxA = new JTextField("1.0", 5);

        //
        canvas = new PiecewiseCanvas();

        //
        updateRange = new JButton("Update");
        updateRange.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                Vector<Double> params = canvas.data.params;
                if (getMaxT() <= 0) {
                    JOptionPane.showMessageDialog(mainPanel,
                            "Stop Time cannot be negative or zero");
                } else if (getMinA() >= getMaxA()) {
                    JOptionPane.showMessageDialog(mainPanel,
                "Minimum Amplitude cannot be greater than Maximum Amplitude");
                } else if (!okToChangeMaxT(params, getMaxT(), canvas.data.maxT,
                                      canvas.data.impliedAmp())) {
                    JOptionPane.showMessageDialog(mainPanel,
                "Stop Time is less than the time of an existing envelope point; delete point(s) to reduce stop time.");
                } else {
                    double maxT = getMaxT();
                    // trim extra points
                    while (params.size() > 0 &&
                           params.get(params.size() - 2) > maxT) {
                        params.removeElementAt(params.size() - 1);
                        params.removeElementAt(params.size() - 1);
                    }
                    // insert new point at maxT if last point is not implied and
                    // if there is not already a point at the new maxT
                    double lastT = (params.size() >= 2 ? 
									params.get(params.size() - 2) : -1);
                    double lastA = (params.size() >= 2 ?
									params.lastElement() : 0);
                    System.out.println("Stretch: lastT " + lastT + " lastA " +
                                       lastA + " data.maxT " + canvas.data.maxT +
                                       " getMaxT() " + getMaxT());
                    if (within(lastT, canvas.data.maxT, EPSILON) &&
                        !within(lastA, canvas.data.impliedAmp(), EPSILON) &&
                        !within(lastT, maxT, EPSILON)) {
                        params.addElement(maxT);
                        params.addElement(lastA);
                    }
					System.out.println("canvas.data.maxT = " + maxT);
                    canvas.data.maxT = maxT;
                    // check for a negative amplitude and warn the user.
                    if (promptNegativeAmp && (getMinA() < 0)) {
                        Object[] options = { "OK", "CANCEL" };
                        int i = JOptionPane.showOptionDialog(mainPanel,
                        "An envelope with negative amplitudes may not produce the expected results.  Make this change anyway?",
                                       "Warning", JOptionPane.DEFAULT_OPTION,
                                        JOptionPane.WARNING_MESSAGE, null,
                                        options, options[0]);
                        if (i == JOptionPane.OK_OPTION)
                            // The user has been warned and wants to
                            // change the range anyway
                            promptNegativeAmp = false;
                        else
                            // The user has told us prevent using the negative
                            // amplitude
                            canvas.restore();
                    }

                    // modified = true; There is no need to do this unless
                    // envelopes are modified to store and load range data
                    canvas.history.save(canvas.data);
                    canvas.envDataToEnvTimes(canvas.data); // refresh the combo box
                    canvas.repaint();
                    return;
                }
                // an error occurred, reset the Range (using complete restore)
                canvas.restore();
            }
        });

        //
        // Used to display extra envelopes in the 'background'
        String[] tmp = { "" };
        compare1 = new JComboBox(tmp);
        compare1.setForeground(Color.red);
        compare1.setPreferredSize(currEnvName.getPreferredSize());
        compare1.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                canvas.repaint();
            }
        });
        compare2 = new JComboBox(tmp);
        compare2.setForeground(new Color(220, 130, 0));// dark yellow
        compare2.setPreferredSize(currEnvName.getPreferredSize());
        compare2.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                canvas.repaint();
            }
        });
        compare3 = new JComboBox(tmp);
        compare3.setForeground(new Color(0, 128, 0)); // green
        compare3.setPreferredSize(currEnvName.getPreferredSize());
        compare3.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                canvas.repaint();
            }
        });
        compare4 = new JComboBox(tmp);
        compare4.setForeground(Color.magenta);
        compare4.setPreferredSize(currEnvName.getPreferredSize());
        compare4.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                canvas.repaint();
            }
        });

        //
        updateGrid = new JButton("Update Grid");
        updateGrid.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                canvas.repaint();
            }
        });

        //
        snap = new JButton("Snap Points to Grid");
        snap.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                Vector<Double> oldparams = (Vector<Double>) 
                        canvas.data.params.clone();
                double tres = getGridX();
                double ares = getGridY();
                boolean save = canvas.mouseDown;
                canvas.mouseDown = true;
                if (canvas.data.params != null)
                    canvas.data.params.removeAllElements();
                envTimes.removeAllItems();
                for (int i = 0; i < oldparams.size(); i += 2) {
                    double time = Math.round(oldparams.get(i) / tres) * tres;
                    double amp = Math.round(oldparams.get(i + 1) / ares) * ares;
                    if (canvas.data.params.contains(time))
                        continue; // consolidate overlapping points
                    canvas.data.addPoint(i, time, amp);
                }
                canvas.mouseDown = false;
                canvas.history.save(canvas.data);
            }
        });

        //
        // components for envelope edit panel
        undo = new JButton("Undo");
        undo.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                canvas.history.undo();
                canvas.restore();
            }
        });
        redo = new JButton("Redo");
        redo.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                canvas.history.redo();
                canvas.restore();
            }
        });
        clear = new JButton("Clear");
        clear.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                System.out.println("calling canvas.clear\n");
                canvas.clear();
            }
        });
        output = new JButton("Output Envelope");
        output.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                boolean isSal = myParent.codeInputPane.isSal;
                String outputStr = canvas.getExpression(isSal);
                codeInput.setText(codeInput.getText().concat(outputStr));
            }
        });
        copy = new JButton("Copy to Clipboard");
        copy.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                boolean isSal = myParent.codeInputPane.isSal;
                StringSelection ss = 
                        new StringSelection(canvas.getExpression(isSal));
                Clipboard clipboard = getToolkit().getSystemClipboard();
                clipboard.setContents(ss, ss);
            }
        });

        //
        timeRes = new JTextField(".1", 5);
        ampRes = new JTextField(".2", 5);
    }

    private void constructComponents() {

        setBounds(100, 100, 500, 503);

        JPanel upperPanel = new JPanel();
        upperPanel.setPreferredSize(new Dimension(1, 200));
        getContentPane().add(upperPanel, BorderLayout.NORTH);
        upperPanel.setLayout(new GridLayout(1, 0, 0, 0));

        JPanel envelopePanel = new JPanel();
        envelopePanel.setBorder(new TitledBorder(null, "Envelope",
                TitledBorder.LEADING, TitledBorder.TOP, null, null));
        upperPanel.add(envelopePanel);
        envelopePanel.setLayout(new BorderLayout(0, 0));

        JPanel nameSubPanel = new JPanel();
        envelopePanel.add(nameSubPanel, BorderLayout.NORTH);
        nameSubPanel.setLayout(new BorderLayout(0, 0));

        saveEnv.setPreferredSize(new Dimension(84, 29));
        nameSubPanel.add(saveEnv, BorderLayout.EAST);

        nameSubPanel.add(currEnvName);

        JLabel lblName = new JLabel("Name");
        nameSubPanel.add(lblName, BorderLayout.WEST);

        JPanel moreEnvSubPanel = new JPanel();
        envelopePanel.add(moreEnvSubPanel, BorderLayout.SOUTH);
        moreEnvSubPanel.setLayout(new BorderLayout(0, 0));

        JPanel panel_6 = new JPanel();
        moreEnvSubPanel.add(panel_6, BorderLayout.CENTER);
        GridBagLayout gbl_panel_6 = new GridBagLayout();
        gbl_panel_6.columnWidths = new int[] { 52, 52, 0 };
        gbl_panel_6.rowHeights = new int[] { 27, 27, 0, 0 };
        gbl_panel_6.columnWeights = new double[] { 1.0, 1.0, Double.MIN_VALUE };
        gbl_panel_6.rowWeights = new double[] { 0.0, 0.0, 0.0, Double.MIN_VALUE };
        panel_6.setLayout(gbl_panel_6);

        JLabel lblMoreEnvelopes = new JLabel("More Envelopes");
        GridBagConstraints gbc_lblMoreEnvelopes = new GridBagConstraints();
        gbc_lblMoreEnvelopes.insets = new Insets(0, 0, 5, 5);
        gbc_lblMoreEnvelopes.gridx = 0;
        gbc_lblMoreEnvelopes.gridy = 0;
        gbc_lblMoreEnvelopes.gridwidth = 2;
        panel_6.add(lblMoreEnvelopes, gbc_lblMoreEnvelopes);

        GridBagConstraints gbc_comboBox_3 = new GridBagConstraints();
        gbc_comboBox_3.fill = GridBagConstraints.HORIZONTAL;
        gbc_comboBox_3.anchor = GridBagConstraints.NORTH;
        gbc_comboBox_3.insets = new Insets(0, 0, 5, 5);
        gbc_comboBox_3.gridx = 0;
        gbc_comboBox_3.gridy = 1;
        panel_6.add(compare1, gbc_comboBox_3);

        GridBagConstraints gbc_comboBox_4 = new GridBagConstraints();
        gbc_comboBox_4.insets = new Insets(0, 0, 5, 0);
        gbc_comboBox_4.fill = GridBagConstraints.HORIZONTAL;
        gbc_comboBox_4.anchor = GridBagConstraints.NORTH;
        gbc_comboBox_4.gridx = 1;
        gbc_comboBox_4.gridy = 1;
        panel_6.add(compare2, gbc_comboBox_4);

        GridBagConstraints gbc_comboBox_1 = new GridBagConstraints();
        gbc_comboBox_1.fill = GridBagConstraints.HORIZONTAL;
        gbc_comboBox_1.anchor = GridBagConstraints.NORTH;
        gbc_comboBox_1.insets = new Insets(0, 0, 0, 5);
        gbc_comboBox_1.gridx = 0;
        gbc_comboBox_1.gridy = 2;
        panel_6.add(compare3, gbc_comboBox_1);

        GridBagConstraints gbc_comboBox_2 = new GridBagConstraints();
        gbc_comboBox_2.fill = GridBagConstraints.HORIZONTAL;
        gbc_comboBox_2.anchor = GridBagConstraints.NORTH;
        gbc_comboBox_2.gridx = 1;
        gbc_comboBox_2.gridy = 2;
        panel_6.add(compare4, gbc_comboBox_2);

        JPanel currAndTypeSubPanel = new JPanel();
        envelopePanel.add(currAndTypeSubPanel, BorderLayout.CENTER);
        currAndTypeSubPanel.setLayout(new BorderLayout(0, 0));

        currAndTypeSubPanel.add(deleteEnv, BorderLayout.EAST);

        currAndTypeSubPanel.add(envName);

        JLabel lblCurrent = new JLabel("->");
        lblCurrent.setHorizontalAlignment(SwingConstants.TRAILING);
        lblCurrent.setPreferredSize(new Dimension(36, 16));
        currAndTypeSubPanel.add(lblCurrent, BorderLayout.WEST);

        JPanel panel_17 = new JPanel();
        currAndTypeSubPanel.add(panel_17, BorderLayout.SOUTH);
        panel_17.setLayout(new BorderLayout(0, 0));

        JLabel lblType = new JLabel("Type");
        lblType.setPreferredSize(new Dimension(36, 16));
        panel_17.add(lblType, BorderLayout.WEST);

        panel_17.add(envTypes, BorderLayout.CENTER);
        
        loadEnv.setPreferredSize(new Dimension(84, 29));
        panel_17.add(loadEnv, BorderLayout.EAST);

        JPanel panel_11 = new JPanel();
        upperPanel.add(panel_11);
        panel_11.setLayout(new BorderLayout(0, 0));

        JPanel panel_2 = new JPanel();
        panel_11.add(panel_2, BorderLayout.NORTH);
        panel_2.setBorder(new TitledBorder(null, "Points",
                TitledBorder.LEADING, TitledBorder.TOP, null, null));
        panel_2.setLayout(new BorderLayout(0, 0));

        JPanel panel_7 = new JPanel();
        panel_2.add(panel_7, BorderLayout.NORTH);
        panel_7.setLayout(new GridLayout(1, 0, 0, 0));

        panel_7.add(addPoint);

        panel_7.add(deletePoint);

        panel_7.add(updatePoint);

        JPanel panel_8 = new JPanel();
        panel_2.add(panel_8, BorderLayout.CENTER);
        GridBagLayout gbl_panel_8 = new GridBagLayout();
        gbl_panel_8.columnWidths = new int[] { 0, 0, 0, 0, 0 };
        gbl_panel_8.rowHeights = new int[] { 0, 0 };
        gbl_panel_8.columnWeights = new double[] { 0.0, 1.0, 0.0, 1.0,
                Double.MIN_VALUE };
        gbl_panel_8.rowWeights = new double[] { 0.0, Double.MIN_VALUE };
        panel_8.setLayout(gbl_panel_8);

        JLabel lblTime = new JLabel("Time");
        GridBagConstraints gbc_lblTime = new GridBagConstraints();
        gbc_lblTime.anchor = GridBagConstraints.EAST;
        gbc_lblTime.insets = new Insets(0, 0, 0, 5);
        gbc_lblTime.gridx = 0;
        gbc_lblTime.gridy = 0;
        panel_8.add(lblTime, gbc_lblTime);

        GridBagConstraints gbc_comboBox_6 = new GridBagConstraints();
        gbc_comboBox_6.insets = new Insets(0, 0, 0, 5);
        gbc_comboBox_6.fill = GridBagConstraints.HORIZONTAL;
        gbc_comboBox_6.gridx = 1;
        gbc_comboBox_6.gridy = 0;
        panel_8.add(envTimes, gbc_comboBox_6);

        JLabel lblAmpl = new JLabel("Ampl");
        GridBagConstraints gbc_lblAmpl = new GridBagConstraints();
        gbc_lblAmpl.insets = new Insets(0, 0, 0, 5);
        gbc_lblAmpl.anchor = GridBagConstraints.ABOVE_BASELINE;
        gbc_lblAmpl.gridx = 2;
        gbc_lblAmpl.gridy = 0;
        panel_8.add(lblAmpl, gbc_lblAmpl);

        GridBagConstraints gbc_spinner = new GridBagConstraints();
        gbc_spinner.fill = GridBagConstraints.HORIZONTAL;
        gbc_spinner.gridx = 3;
        gbc_spinner.gridy = 0;
        panel_8.add(envAmplitudes, gbc_spinner);

        JPanel panel_12 = new JPanel();
        panel_11.add(panel_12, BorderLayout.CENTER);
        panel_12.setLayout(new BorderLayout(0, 0));

        output.setText("Output Envelope");

        JPanel panel_13 = new JPanel();
        panel_12.add(panel_13, BorderLayout.CENTER);
        GridBagLayout gbl_panel_13 = new GridBagLayout();
        panel_13.setLayout(gbl_panel_13);

        GridBagConstraints gbc_output = new GridBagConstraints();
        gbc_output.anchor = GridBagConstraints.NORTH;
        gbc_output.gridx = 0;
        gbc_output.gridy = 0;
        panel_13.add(output, gbc_output);

        dispCoord = new JToggleButton("Show Coordinates");
        dispCoord.addActionListener(new ActionListener() {
                @Override
                    public void actionPerformed(ActionEvent e) {
                    displayCoord = dispCoord.isSelected();
                    canvas.repaint();                
                }
            });
        
        GridBagConstraints gbc_dispCoord = new GridBagConstraints();
        gbc_dispCoord.anchor = GridBagConstraints.NORTH;
        gbc_dispCoord.gridx = 0;
        gbc_dispCoord.gridy = 1;
        panel_13.add(dispCoord, gbc_dispCoord);
        
        //
        toggleGrid = new JToggleButton("Show Grid", false);
        toggleGrid.addActionListener(new ActionListener() {
                @Override
                    public void actionPerformed(ActionEvent arg0) {
                    canvas.repaint();
                }
            });
        GridBagConstraints gbc_toggleGrid = new GridBagConstraints();
        gbc_toggleGrid.anchor = GridBagConstraints.NORTH;
        gbc_toggleGrid.gridx = 0;
        gbc_toggleGrid.gridy = 2;
        panel_13.add(toggleGrid, gbc_toggleGrid);

        GridBagConstraints gbc_copy = new GridBagConstraints();
        gbc_copy.anchor = GridBagConstraints.NORTH;
        gbc_copy.gridx = 0;
        gbc_copy.gridy = 3;
        panel_13.add(copy, gbc_copy);
        
        JPanel panel_9 = new JPanel();
        panel_9.setBorder(new TitledBorder(null, "Envelope View",
                TitledBorder.LEADING, TitledBorder.TOP, null, null));
        getContentPane().add(panel_9, BorderLayout.CENTER);
        panel_9.setLayout(new BorderLayout(0, 0));

        JPanel panel_10 = new JPanel();
        panel_9.add(panel_10, BorderLayout.NORTH);

        panel_10.add(undo);

        panel_10.add(redo);

        panel_10.add(clear);

        JPanel panel_14 = new JPanel();
        panel_9.add(panel_14, BorderLayout.SOUTH);
        GridBagLayout gbl_panel_14 = new GridBagLayout();
        gbl_panel_14.columnWidths = new int[] { 55, 37, 61, 37, 121, 0 };
        gbl_panel_14.rowHeights = new int[] { 29, 0 };
        gbl_panel_14.columnWeights = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0,
                Double.MIN_VALUE };
        gbl_panel_14.rowWeights = new double[] { 0.0, Double.MIN_VALUE };
        panel_14.setLayout(gbl_panel_14);

        JLabel lblGridSize = new JLabel("Grid-x Size");
        GridBagConstraints gbc_lblGridSize = new GridBagConstraints();
        gbc_lblGridSize.anchor = GridBagConstraints.WEST;
        gbc_lblGridSize.insets = new Insets(0, 0, 0, 5);
        gbc_lblGridSize.gridx = 0;
        gbc_lblGridSize.gridy = 0;
        panel_14.add(lblGridSize, gbc_lblGridSize);

        GridBagConstraints gbc_spinner_1 = new GridBagConstraints();
        gbc_spinner_1.anchor = GridBagConstraints.NORTHWEST;
        gbc_spinner_1.insets = new Insets(0, 0, 0, 5);
        gbc_spinner_1.gridx = 1;
        gbc_spinner_1.gridy = 0;
        panel_14.add(timeRes, gbc_spinner_1);

        JLabel lblMaxTime = new JLabel("Max Time");
        GridBagConstraints gbc_lblMaxTime = new GridBagConstraints();
        gbc_lblMaxTime.anchor = GridBagConstraints.WEST;
        gbc_lblMaxTime.insets = new Insets(0, 0, 0, 5);
        gbc_lblMaxTime.gridx = 2;
        gbc_lblMaxTime.gridy = 0;
        panel_14.add(lblMaxTime, gbc_lblMaxTime);

        GridBagConstraints gbc_spinner_2 = new GridBagConstraints();
        gbc_spinner_2.anchor = GridBagConstraints.NORTHWEST;
        gbc_spinner_2.insets = new Insets(0, 0, 0, 5);
        gbc_spinner_2.gridx = 3;
        gbc_spinner_2.gridy = 0;
        panel_14.add(maxT, gbc_spinner_2);

        GridBagConstraints gbc_btnSnapToGrid = new GridBagConstraints();
        gbc_btnSnapToGrid.anchor = GridBagConstraints.NORTHWEST;
        gbc_btnSnapToGrid.gridx = 4;
        gbc_btnSnapToGrid.gridy = 0;
        panel_14.add(snap, gbc_btnSnapToGrid);

        JPanel panel_15 = new JPanel();
        panel_9.add(panel_15, BorderLayout.EAST);
        GridBagLayout gbl_panel_15 = new GridBagLayout();
        gbl_panel_15.columnWidths = new int[] { 34, 0 };
        gbl_panel_15.rowHeights = new int[] { 16, 0, 0, 0, 0, 0 };
        gbl_panel_15.columnWeights = new double[] { 0.0, Double.MIN_VALUE };
        gbl_panel_15.rowWeights = new double[] { 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0 };
        panel_15.setLayout(gbl_panel_15);

        JLabel lblResol = new JLabel("Grid-y size");
        GridBagConstraints gbc_lblResol = new GridBagConstraints();
        gbc_lblResol.fill = GridBagConstraints.HORIZONTAL;
        gbc_lblResol.insets = new Insets(0, 0, 5, 0);
        gbc_lblResol.anchor = GridBagConstraints.NORTH;
        gbc_lblResol.gridx = 0;
        gbc_lblResol.gridy = 0;
        panel_15.add(lblResol, gbc_lblResol);

        GridBagConstraints gbc_spinner_3 = new GridBagConstraints();
        gbc_spinner_3.fill = GridBagConstraints.HORIZONTAL;
        gbc_spinner_3.insets = new Insets(0, 0, 5, 0);
        gbc_spinner_3.gridx = 0;
        gbc_spinner_3.gridy = 1;
        panel_15.add(ampRes, gbc_spinner_3);

        JLabel lblRange = new JLabel("Min~Max");
        GridBagConstraints gbc_lblRange = new GridBagConstraints();
        gbc_lblRange.fill = GridBagConstraints.HORIZONTAL;
        gbc_lblRange.insets = new Insets(0, 0, 5, 0);
        gbc_lblRange.gridx = 0;
        gbc_lblRange.gridy = 2;
        panel_15.add(lblRange, gbc_lblRange);

        GridBagConstraints gbc_spinner_4 = new GridBagConstraints();
        gbc_spinner_4.fill = GridBagConstraints.HORIZONTAL;
        gbc_spinner_4.insets = new Insets(0, 0, 5, 0);
        gbc_spinner_4.gridx = 0;
        gbc_spinner_4.gridy = 3;
        panel_15.add(minA, gbc_spinner_4);

        GridBagConstraints gbc_spinner_5 = new GridBagConstraints();
        gbc_spinner_5.fill = GridBagConstraints.HORIZONTAL;
        gbc_spinner_5.insets = new Insets(0, 0, 5, 0);
        gbc_spinner_5.gridx = 0;
        gbc_spinner_5.gridy = 4;
        panel_15.add(maxA, gbc_spinner_5);
        
        updateRange.setPreferredSize(new Dimension(72, 29));
        GridBagConstraints gbc_btnUpdate = new GridBagConstraints();
        gbc_btnUpdate.gridx = 0;
        gbc_btnUpdate.gridy = 5;
        panel_15.add(updateRange, gbc_btnUpdate);

        panel_9.add(canvas, BorderLayout.CENTER);
    }

    public void envNameSelected() {
        if (saving)
            return; // ignore selection generated by "save" button
        // If the name is different from the current envelope name, do
        // a "save". Then switch to editing the newly selected envelope.
        String name = (String) envName.getSelectedItem();
        // when we load the JComboBox with new names, the contentsChanged
        // method of JComboBox invokes a selection action, even if nothing
        // is selected. null occures when every envelope is deleted.
        if (name == null) {
            canvas.clear();
            return;
        }

        String originalName = currentEnvName;
        currentEnvName = currEnvName.getText().trim();
        if (!originalName.equals(currentEnvName)) {
            modified = true;
        }
        if (modified && !deleting) {
            Object[] options = { "OK", "CANCEL" };
            int i = JOptionPane.showOptionDialog(mainPanel, currentEnvName
                    + " is being edited. Save it?", "Warning",
                    JOptionPane.DEFAULT_OPTION, JOptionPane.WARNING_MESSAGE,
                    null, options, options[0]);
            if (i == JOptionPane.OK_OPTION) {
                saveEnvelopes();
            }
        }

        // store envelope under old name in XLISP syntax
        String edited = canvas.getExpression(false);
        canvas.clear();
        System.out.println("envNameSelected: expression existed, putting " + currentEnvName
                + " " + edited + ", changing currentEnvName to " + name);
        if (currentEnvName.length() > 0)
            envColl.put(currentEnvName, edited);
        // install name as new envelope to edit
        String expression = envColl.get(name);
        System.out.println("envNameSelected: new name " + name + 
                           " expression is " + expression);
        // set the linear/exponential selector according to the expression
        canvas.data.linear = canvas.isEnvDataLinear(expression);
        envTypes.setSelectedItem(types[canvas.data.linear ? PWL_TYPE 
                                                          : PWE_TYPE]);
        canvas.setEnv(expression);
        currEnvName.setText(name);
        // make name be the selected name
        envName.setSelectedItem(name);
        currentEnvName = name;
    }

    // public double getMinT() { return
    // Double.parseDouble(minT.getText().trim()); }
    public double getMaxT() {
        return Double.parseDouble(maxT.getText().trim());
    }

    public double getMinA() {
        return Double.parseDouble(minA.getText().trim());
    }

    public double getMaxA() {
        return Double.parseDouble(maxA.getText().trim());
    }

    public double getGridX() {
        return Double.parseDouble(timeRes.getText().trim());
    }

    public double getGridY() {
        return Double.parseDouble(ampRes.getText().trim());
    }

    public boolean getGridToggle() {
        return toggleGrid.isSelected();
    }

    public boolean getEnvLinear() {
        if (envTypes != null) {
            String env = (String) envTypes.getSelectedItem();
            return env.matches(types[PWL_TYPE]);
        }
        return true; // default when initializing
    }

    public boolean within(double x, double y, double eps) {
        return Math.abs(x - y) < eps;
    }

    // write current envelope definition to Nyquist
    public void saveEnvelopes() {
        // make sure current envelope has been stored in collection
        if (currEnvName.getText().length() == 0) {
            JOptionPane.showMessageDialog(mainPanel,
                    "Please Enter an Envelope Name");
            return;
        }
        currentEnvName = currEnvName.getText().trim();
        // now write all to Nyquist in XLISP format
        saving = true;
        boolean foundIt = false;
        for (Enumeration keys = envColl.keys(); keys.hasMoreElements();) {
            String name = (String) keys.nextElement();
            // update envelope collection with current envelope
            if (name.equals(currentEnvName)) {
                envColl.remove(name);
                envColl.put(currentEnvName, canvas.getExpression(false));
                foundIt = true;
            }
            String expression = envColl.get(name);
            String defn = "(define-env '" + name + " '" + expression + ")";
            System.out.print("in saveEnvelopes: " + defn);
            myParent.sendInputLn(defn); // send to Nyquist for evaluation
        }
        // if the current envelope was not in the list, add it and save it
        if (!foundIt) {
            String expr = canvas.getExpression(false);
            envColl.put(currentEnvName, expr);
            envName.addItem(currentEnvName);
            compare1.addItem(currentEnvName);
            compare2.addItem(currentEnvName);
            compare3.addItem(currentEnvName);
            compare4.addItem(currentEnvName);
            String defn = "(define-env '" + currentEnvName + " '" + expr + ")";
            System.out.print("in saveEnvelopes: " + defn);
            myParent.sendInputLn(defn); // send to Nyquist for evaluation
        }
        envName.setSelectedItem(currentEnvName);
        modified = false;
        saving = false;
        System.out.println("modified set to false in saveEnvelopes\n");
    }

    public void loadEnvelopes() {
        myParent.callFunction("get-env-data", "");
    }

    public void deleteEnvelope() {
        Object[] options = { "OK", "CANCEL" };
        int i = JOptionPane.showOptionDialog(mainPanel,
                "Deletion cannot be undone, click OK to continue", "Warning",
                JOptionPane.DEFAULT_OPTION, JOptionPane.WARNING_MESSAGE, null,
                options, options[0]);
        if (i != JOptionPane.OK_OPTION)
            return;
        deleting = true; // without this, a user may be prompted to save the
                            // envelope they just deleted if it has been edited.
        // it appears as though currentEnvName gets changed when you remove
        // it from the envName comboBox, so make a local copy. Previously,
        // we were calling DELETE-ENV after the name changed!
        String name = currentEnvName;
        envColl.remove(name); // delete the envelope from hashtable
        envName.removeItem(name); // delete the envelope from the combobox
        // delete the envelope from the comparison comboboxes
        if (name.equals(compare1.getSelectedItem()))
            compare1.setSelectedItem("");
        if (name.equals(compare2.getSelectedItem()))
            compare2.setSelectedItem("");
        if (name.equals(compare3.getSelectedItem()))
            compare3.setSelectedItem("");
        if (name.equals(compare4.getSelectedItem()))
            compare4.setSelectedItem("");
        compare1.removeItem(name);
        compare2.removeItem(name);
        compare3.removeItem(name);
        compare4.removeItem(name);
        myParent.sendInputLn("(delete-env '" + name + ")"); // delete the
                                                            // envelope from the
                                                            // workspace
        deleting = false;
    }

    public void actionPerformed(ActionEvent e) {
        // System.out.println(e.getActionCommand());
        String actionCommand = e.getActionCommand();

        // File Menu options
        if (actionCommand.equals("saveEnvelopes")) {
            saveEnvelopes();
        } else if (actionCommand.equals("loadEnvelope")) {
            loadEnvelopes();
        } else if (actionCommand.equals("deleteEnvelope")) {
            deleteEnvelope();
        } else if (actionCommand.equals("envNameSelected")) {
            envNameSelected();
        } else if (actionCommand.equals("envTypeChanged")) {
            System.out.println("envTypechanged action raised");
            boolean isLinear = getEnvLinear();
            if (isLinear != canvas.data.linear) {
                canvas.data.linear = isLinear;
                canvas.history.save(canvas.data);
                // canvas.repaint(); -- redundant
            }

            // I'm not sure why this was here. It seems like the
            // default range should always be [0,1] even though
            // real zero is a problem for exponential envelopes.
            // set initial amplitude and time parameters
            //            if (canvas.times.size() < 1) {
            //                if (type == PWE_TYPE) {
            //                    minA.setText("1.0");
            //                    maxA.setText("2.0");
            //                } else {
            //                    minA.setText("0.0");
            //                    maxA.setText("1.0");
            //                }
            //            }
            canvas.repaint();
            validate();
            setVisible(true);
            System.out.println("envTypechanged returns, canvas.data.linear " +
                               canvas.data.linear);
        } else if (actionCommand.equals("deletePoint")) {
            int index = envTimes.getSelectedIndex();
            System.out.println("at deletePoint, index " + index);
            if (index >= 0) {
                canvas.selection = index * 2;
                System.out.println("at deletePoint before deleteSelection");
                canvas.deleteSelection();
                index = envTimes.getSelectedIndex();
                System.out.println("deletePoint new index " + index);
                if (index >= 0) {
                    canvas.selection = index * 2;
                    envAmplitudes.setText(
                            form.format(canvas.data.params.get(
                                                       canvas.selection)));
                }
                canvas.repaint();
            }
        } else if (actionCommand.equals("addPoint")) {
            String text = (String) envTimes.getSelectedItem();
            if (text.equals(""))
                return;
            double time = Double.parseDouble(text.trim());
            text = envAmplitudes.getText();
            double amp = Double.parseDouble(text.trim());
            canvas.selection = canvas.data.insertInOrder(time, amp);
            canvas.changed = true;
            canvas.history.save(canvas.data);
            canvas.repaint();
        } else if (actionCommand.equals("updatePoint")) {
            String text = (String) envTimes.getSelectedItem();
            if (text.equals(""))
                return;
            double time = Double.parseDouble(text.trim());
            text = envAmplitudes.getText();
            double amp = Double.parseDouble(text.trim());
            System.out.println("updatePoint selection " + canvas.selection);
            canvas.deleteSelection();
            canvas.selection = canvas.data.insertInOrder(time, amp);
            canvas.changed = true;
            canvas.history.save(canvas.data);
            canvas.repaint();
        } else if (actionCommand.equals("envTimeChange")) {
            // sometimes this action gets called in the middle of
            // doing an update and in an inconsistent state. If this
            // happens, don't try to set amplitude text.
            System.out.println("envTimeChange " + envTimes.getSelectedIndex() +
                               " items " + envTimes.getItemCount());
            if (canvas.data.params.size() / 2 != envTimes.getItemCount())
                return;
            int index = envTimes.getSelectedIndex();
            if (index >= 0) {
                canvas.selection = index * 2;
                envAmplitudes.setText(
                    form.format(canvas.data.params.get(canvas.selection + 1)));
                canvas.repaint(); // update selection marker
            }
        } else {
            System.out.println("ACTION NOT HANDLED: " + actionCommand);
        }
    }

    public void loadEnvData(String data) {
        data = data.toLowerCase();
        System.out.println("loadEnvData: data |" + data + "| len "
                + data.length());
        envColl.clear();
        envName.removeAllItems(); // clear and reload combo box
        envTimes.removeAllItems(); // clear times
        envAmplitudes.setText(""); // clear amplitude box
        compare1.removeAllItems(); // reset the comparison combo boxes
        compare1.addItem("");
        compare2.removeAllItems();
        compare2.addItem("");
        compare3.removeAllItems();
        compare3.addItem("");
        compare4.removeAllItems();
        compare4.addItem("");
        while (data.length() > 0) {
            int eolx = data.indexOf("\n");
            if (eolx < 0) // shouldn't happen, but bail if it does
                return;
            String line = data.substring(0, eolx);
            System.out.println("loadEnvData: line " + line);
            data = data.substring(eolx + 1);
            String name = line.substring(0, line.indexOf(' '));
            System.out.println("loadEnvData: name " + name);
            String env = line.substring(name.length() + 1);
            System.out.println("loadEnvData: env " + env);
            if (name.length() > 0)
                envColl.put(name, env);
            envName.addItem(name);
            compare1.addItem(name);
            compare2.addItem(name);
            compare3.addItem(name);
            compare4.addItem(name);
        }
    }

    private class EnvData {
        public Vector<Double> params; // alternating times and amplitudes
        public boolean linear; // interpolation is linear 
        public double maxT;    // the maximum time
        public double minA;    // the minimum amplitude to plot
        public double maxA;    // the maximum amplitude to plot

        public EnvData() {
            params = new Vector<Double>();
            linear = true;
            maxT = 1.0;
            minA = 0.0;
            maxA = 1.0;
        }

        public EnvData(EnvData data) {
            params = (Vector<Double>) data.params.clone();
            linear = data.linear;
            maxT = data.maxT;
            minA = data.minA;
            maxA = data.maxA;
        }

        
        public boolean isImpliedFirstPoint() {
            return (params.size() == 0 || 
                    !within(params.get(0), 0, EPSILON));
        }


        public boolean isImpliedLastPoint() {
            return (params.size() == 0 ||
                    !within(params.get(params.size() - 2), maxT, EPSILON));
        }


        public double impliedAmp() {
            return (linear ? 0.0 : 1.0);
        }


        public int addPoint(int i, double absT, double absA) {
            System.out.println("addPoint: " + i + " " + absT + " " + absA);
            params.add(i, absA);
            params.add(i, absT); // order in params will be (absT, absA)
            envTimes.insertItemAt(form.format(absT), i / 2);
            System.out.println("addPoint time: " + absT + ", text "
                    + form.format(absT));
            envTimes.setSelectedIndex(i / 2);
            envAmplitudes.setText(form.format(params.get(i + 1)));
            return i;
        }

        // insert the time and amplitude in params in time-sorted order
        public int insertInOrder(double time, double amp) {
            int i = 0;
            modified = true;
            if (params != null) {
                while (i < params.size() && time > params.get(i)) {
                    i += 2;
                }
            }
            addPoint(i, time, amp);
            // if there are now two points at the origin or the end, delete
            // the other point
            if (params.size() > 2) {
                if (within(time, 0.0, EPSILON)) {
                    // the "other" point is at index 0 or 2
                    int j = 0;
                    if (i == j) j = 2;
                    if (within(params.get(j), 0.0, EPSILON)) {
                        params.removeElementAt(j);
                        params.removeElementAt(j);
                        // i should now be the first point:
                        i = 0;
                    }
                } else if (within(time, maxT, EPSILON)) {
                    int j = params.size() - 2;
                    if (j == i) j -= 2;
                    if (within(params.get(j), maxT, EPSILON)) {
                        params.removeElementAt(j);
                        params.removeElementAt(j);
                        // i should now be the last point
                        i = params.size() - 2;
                    }
                }
            }
            return i;
        }
    }


    private class History {
        /*
         * consider a sequence of undo/redo to be a single edit operation --
         * thus the end of the versions list is set to the latest undo/redo
         * selection
         */
        private boolean undoRedo = false; // flag so that actions are duplicated
                                            // as noted above
        private Vector<EnvData> history = new Vector<EnvData>();
        private int version = -1;
        // We need a flag to know when changes have been made. The version
        // variable
        // is changed when the user hits undo/redo. Without the flag, we can hit
        // edge cases where, for example, hitting redo will save the state
        // again.
        // The user would then have to press undo twice in a row to revert said
        // state.
        private int versionMax = -1;

        public void save(EnvData data) {
            EnvData envData = new EnvData(data);
            history.add(envData);
            version = history.size() - 1;
            versionMax = version;
            System.out.println("Saved version " + version);
            undoRedo = false; // clear flag for next undoRedo sequence
        }

        public boolean canGet() {
            boolean result = version >= 0 && version < history.size();
            System.out.println("canGet returns " + result + " version " + 
                               version);
            return result;
        }

        public EnvData getData() {
            return history.get(version);
        }

        private void processUndoRedo() {
            if (!undoRedo) { // extend with copy of the version
                history.add(new EnvData(history.get(version)));
            } else { // replace with different version
                history.set(history.size() - 1, 
                            new EnvData(history.get(version)));
            }
            undoRedo = true;
        }

        public void undo() {
            if (version > 0) {
                version--;
                processUndoRedo();
            }
        }

        public void redo() {
            if ((version != versionMax) && (version < history.size() - 1)) {
                version++;
                processUndoRedo();
            }
        }
    }

    // Class for the drawing area
    // private class PiecewiseCanvas extends Canvas implements MouseListener,
    private class PiecewiseCanvas extends JPanel implements MouseListener,
            MouseMotionListener, KeyListener {
        private int currentAmp;
        private int currentTime;
        public BufferedImage image = null;
        public boolean selectCheck = false;
        public int selection;
        public History history;
        private boolean mouseDown = false; // used to detect version for undo
        private boolean changed = false; // used to detect version for undo

        public EnvData data = new EnvData();

        // Constructor
        public PiecewiseCanvas() {
            setBackground(Color.white);
            addMouseListener(this);
            addMouseMotionListener(this);
            addKeyListener(this);
            selection = -1;
            history = new History();
            history.save(data);
        }


        // Allow JPanel to accept keyboard events
        public boolean isFocusable() {
            return true;
        }

        // try to make the canvas the correct size
        public Dimension getMinimumSize() {
            return new Dimension(0, 0);
        }

        public Dimension getPreferredSize() {
            return new Dimension(575, 256);
        }

        // draw the graphics inside the full canvas, so use these
        // functions to get the size and coordinates of the drawing
        // area that is usable
        private int graphLeft() {
            return LEFT_BORDER;
        }

        private int graphRight() {
            return getWidth() - RIGHT_BORDER;
        }

        private int graphWidth() {
            return getWidth() - (LEFT_BORDER + RIGHT_BORDER);
        }

        private int graphTop() {
            return TOP_BORDER;
        }

        private int graphBottom() {
            return getHeight() - BOTTOM_BORDER;
        }

        private int graphHeight() {
            return getHeight() - (TOP_BORDER + BOTTOM_BORDER);
        }

        private int clipX(int x) {
            return Math.max(LEFT_BORDER, Math.min(graphRight(), x));
        }

        private int clipY(int y) {
            return Math.max(TOP_BORDER, Math.min(graphBottom(), y));
        }


        public String getExpression(boolean salFlag) {
            String outputStr;
            int start = 0;
            boolean vType = (!data.isImpliedFirstPoint() || 
                             !data.isImpliedLastPoint());
            String fn = (!data.linear ? "pwev" :
                         (vType ? "pwlv" : "pwl"));
            outputStr = (salFlag ? fn + "(" : "(" + fn + " ");
            String spacer = (salFlag ? ", " : " ");
            boolean linear = isEnvDataLinear(outputStr);
            if (vType) { // insert initial value
                if (data.isImpliedFirstPoint()) {
                    outputStr += form.format(data.impliedAmp());
                } else {
                    outputStr += form.format(data.params.get(1));
                    start = 2;
                }
                outputStr += spacer;
            }

            for (int i = start; i < data.params.size(); i += 2) {
                double time = data.params.get(i);
                double amp = data.params.get(i + 1);
                outputStr += form.format(time) + spacer + 
                             form.format(amp) + spacer;
            }

            if (vType) { // we're already ending with a value
                if (!data.isImpliedLastPoint()) {
                    // we're done because we output at maxT
                    // but there's an extra spacer at the end
                    outputStr = outputStr.substring(0, outputStr.length() - 
                                                          spacer.length());
                } else {
                    outputStr += form.format(data.maxT) + spacer +
                                 form.format(data.impliedAmp());
                }
            } else {
                outputStr += form.format(data.maxT);
            }
            outputStr += ")";
            return outputStr;
        }

        // draw the canvas image
        public void paint(Graphics g) {
            super.paint(g);
            // test: g.drawLine(0, 0, 100, 100);
            Graphics2D drawArea = (Graphics2D) g;

            String name = (String) compare1.getSelectedItem();
            if (name != "")
                drawCompareEnv(drawArea, name, Color.red);
            name = (String) compare2.getSelectedItem();
            if (name != "")
                drawCompareEnv(drawArea, name, new Color(220, 130, 0));
            name = (String) compare3.getSelectedItem();
            if (name != "")
                drawCompareEnv(drawArea, name, new Color(0, 128, 0));
            name = (String) compare4.getSelectedItem();
            if (name != "")
                drawCompareEnv(drawArea, name, Color.magenta);

            canvas.drawGrid(drawArea);
            canvas.drawCircles(drawArea, data, true);
            System.out.println("paint linear " + data.linear);
            connectTheDots(drawArea, data, Color.blue);
            drawSelectionCircle(drawArea);
            drawArea.dispose();
        }


        // draw a comparison envelope
        private void drawCompareEnv(Graphics2D drawArea, String name,
                Color color) {
            EnvData envData = new EnvData();
            parseEnvData(envColl.get(name), envData);
            drawCircles(drawArea, envData, false);
            connectTheDots(drawArea, envData, color);
        }


        // draw a grid if the toggleGrid checkbox is true
        private void drawGrid(Graphics2D g) {
            if (getGridToggle()) {
                int blocks = (int) (getMaxT() / getGridX());
                int maxx = time2x(getMaxT());
                int maxy = amp2y(getMinA());
                g.setColor(Color.lightGray);
                for (int x = 0; x <= blocks; x++) {
                    int loc = time2x(x * getGridX());
                    g.drawLine(loc, 0, loc, maxy);
                }
                blocks = (int) ((getMaxA() - getMinA()) / getGridY());
                for (int y = 0; y <= blocks; y++) {
                    int loc = amp2y(y * getGridY());
                    g.drawLine(0, loc, maxx, loc);
                }
            }
        }


        // draw all of the points in the envelope
        private void drawCircles(Graphics2D g, EnvData envData, 
                                 boolean mainEnv) {
            double maxTime = getMaxT();
            int index = 0;

            // The initial point is (0,0) if !valueType
            if (envData.isImpliedFirstPoint()) {
                double amp = envData.impliedAmp();
                int y = amp2y(amp);
                drawDot(g, graphLeft(), y);
                if (mainEnv && displayCoord) {
                    g.drawString("(0, " + amp + ")", graphLeft(), y - 3);
                }
            }

            // is the final point implicit?
            if (envData.isImpliedLastPoint()) {
                double amp = envData.impliedAmp();
                int y = amp2y(amp);
                drawDot(g, graphRight(), y);
                if (mainEnv && displayCoord) {
                    g.drawString(
                            "(" + form.format(envData.maxT) + ", " + amp + ")",
                            graphWidth() - 36, y - 9);
                }
            }

            // draw points for each point in the envelope
            while (index < envData.params.size()) {
                double time = envData.params.get(index);
                int t = time2x(time);
                double amp = envData.params.get(index + 1);
                int a = amp2y(amp);
                if (mainEnv && displayCoord)
                    g.drawString("(" + form.format(time) + ","
                            + form.format(amp) + ")", t, a);
                drawDot(g, t, a);
                // System.out.println("drawDot t " + t + " a " + a + " width " +
                // getWidth());
                index += 2;
            }
        }


        // given coordinates, draw a circle on the canvas
        private void drawDot(Graphics2D g, int t, int a) {
            // draw a black circle at the specified point
            g.setColor(Color.black);
            // System.out.println("drawDot: " + t + "," + a);
            g.fillOval(t - 2, a - 2, 5, 5);
        }

        // given coordinates, draw a circle around selected envelope point
        private void drawSelectionCircle(Graphics2D g) {
            if (selection >= 0 && selection < data.params.size() - 1) {
                int t = time2x(data.params.get(selection));
                int a = amp2y(data.params.get(selection + 1));
                // draw a red circle around the specified point
                g.setColor(Color.red);
                g.drawOval(t - 4, a - 4, 9, 9);
            }
        }

        private void draw_connect(Graphics2D g, double t1, double a1,
                                  double t2, double a2, boolean linear) {
            int x1 = time2x(t1);
            int x2 = time2x(t2);
            int y1 = amp2y(a1);
            int y2 = amp2y(a2);
            if (linear) {
                g.drawLine(x1, y1, x2, y2);
            } else {
                // pwe type, graph is linear along a log scale
                if (a1 <= EPSILON || a2 <= EPSILON) {
                    g.drawLine(x1, y1, x1, graphBottom());
                    g.drawLine(x1, graphBottom(), x2, graphBottom());
                    g.drawLine(x2, graphBottom(), x2, y2);
                } else {
                    double log1 = Math.log(a1);
                    double log2 = Math.log(a2);
                    int startline = y1;
                    double logIncr = (log2 - log1) / (x2 - x1);
                    for (int j = x1 + 1; j <= x2; j++) {
                        double loga = log1 + logIncr * (j - x1);
                        int a = amp2y(Math.exp(loga));
                        g.drawLine(j - 1, startline, j, a);
                        startline = a;
                    }
                }
            }
        }

        // connect adjacent points in the envelope by lines (pwl, pwlv) or by an
        // exponential curve (pwe, pwev)
        private void connectTheDots(Graphics2D g, EnvData data, Color color) {
            g.setColor(color);
            // System.out.println("connectTheDots\n");
            if (data.params.size() > 0) {
                if (data.isImpliedFirstPoint()) {
                    draw_connect(g, 0, data.impliedAmp(), 
                                 data.params.get(0), data.params.get(1), 
                                 data.linear);
                }
                if (data.isImpliedLastPoint()) {
                    int i = data.params.size() - 2;
					System.out.println("data.maxT = " + data.maxT);
                    draw_connect(g, data.params.get(i), data.params.get(i + 1),
                                 data.maxT, data.impliedAmp(), 
                                 data.linear);
                }
                // connect the non-endpoints in the envelope
                double t1 = data.params.get(0);
                double a1 = data.params.get(1);
                for (int i = 2; i < data.params.size(); i += 2) {
                    double t2 = data.params.get(i);
                    double a2 = data.params.get(i + 1);
                    draw_connect(g, t1, a1, t2, a2, data.linear);
                    t1 = t2;
                    a1 = a2;
                }
            } else {
                // size == 0, so both points are implied
                draw_connect(g, 0, data.impliedAmp(), data.maxT, 
                             data.impliedAmp(), data.linear);
            }
        }

        // erase the canvas and clear the parameter vectors -
        // completely reset the envelope
        public void clear() {
            if (data.params != null)
                data.params.removeAllElements();
            envTimes.removeAllItems();
            envAmplitudes.setText("");
            modified = true;
            repaint();
            history.save(data);
        }

        // refresh envTimes combo box from actual data
        public void envDataToEnvTimes(EnvData data) {
            envTimes.removeAllItems();
            for (int i = 0; i < data.params.size(); i += 2) {
                envTimes.insertItemAt(form.format(data.params.get(i)), i / 2);
            }
            envAmplitudes.setText("");            
        }


        public void restore() {
            if (history.canGet()) {
                data = new EnvData(history.getData());
                envTypes.setSelectedItem(types[data.linear ? PWL_TYPE 
                                                           : PWE_TYPE]);
                maxT.setText(form.format(data.maxT));
                maxA.setText(form.format(data.maxA));
                minA.setText(form.format(data.minA));

                selection = -1;
                envDataToEnvTimes(data);
                repaint();
            }
        }

        // set time and amplitude on click by inserting the point into the
        // vectors.
        // if delete is checked, try to delete a point from the envelope
        public void mousePressed(MouseEvent e) {
            mouseDown = true;
            System.out.println("mouseDown true\n");
            this.requestFocus();
            currentTime = e.getX();
            currentAmp = e.getY();
            selectCheck = checkSelection(currentTime, currentAmp);
            if (selectCheck)
                return;
            selection = data.insertInOrder(x2time(currentTime), y2amp(currentAmp));
            changed = true;
            repaint();
        }

        public void mouseDragged(MouseEvent e) {
            currentTime = clipX(e.getX());
            currentAmp = clipY(e.getY());
            if (currentTime < graphRight() && currentTime >= graphLeft()
                    && currentAmp >= graphTop() && currentAmp < graphBottom()) {
                deleteSelection();
                selection = data.insertInOrder(x2time(currentTime), 
                                               y2amp(currentAmp));
                changed = true;
                repaint();
            }
        }

        public void mouseReleased(MouseEvent e) {
            System.out.println("mouseReleased\n");
            if (changed) {
                history.save(data);
                changed = false;
            }
            mouseDown = false;
            System.out.println("mouseDown false\n");
        }

        // convert time coordinate to time in seconds
        private double x2time(int x) {
			double maxTime = getMaxT();
            return (x - graphLeft()) * maxTime / (graphWidth() - 1);
        }

        private int time2x(double time) {
			double maxTime = getMaxT();
            int x = (int) Math.round(time * (graphWidth() / maxTime))
                    + graphLeft();
            return x;
        }

        // convert amp coordinate to real amplitude
        private double y2amp(int y) {
            double maxAmp = getMaxA();
            double aRange = maxAmp - getMinA();
            double amp = maxAmp - ((y - graphTop()) * 
                                   aRange / (graphHeight() - 1));
            System.out.println("y2amp: " + y + " top " + graphTop() + " height " + graphHeight());
            return amp;
        }

        private int amp2y(double amp) {
            double maxAmp = getMaxA();
            double aRange = maxAmp - getMinA();
            int y = (int) Math.round((maxAmp - amp) * graphHeight() / aRange)
                    + graphTop();
            return y;
        }

        private void deleteSelection() {
            if (selection < 0 || selection >= data.params.size())
                return;
            data.params.removeElementAt(selection);
            data.params.removeElementAt(selection);
            modified = true;
            System.out.println("deleteSelection at " + selection);
            envTimes.removeItemAt(selection / 2);
            // make the Amp: box correspond to the new selection:
            String amp = "";
            if (data.params.size() > 0) {
                int index = envTimes.getSelectedIndex();
                if (index >= 0)
                    amp = form.format(data.params.get(index * 2 + 1));
            }
            envAmplitudes.setText(amp);
            selection = -1;
            if (!mouseDown)
                history.save(data);
        }


        // Check if mouse click corresponds to existing envelope point
        // return index of point or -1 if no point is close
        private int getSelection(int x, int y) {
            int cutoff = 7;
            int bestDist = cutoff * 2;
            int bestIndex = -1;
            if (data.params == null)
                return bestIndex;
            for (int i = 0; i < data.params.size(); i += 2) {
                int xi = time2x(data.params.get(i));
                int yi = amp2y(data.params.get(i + 1));
                int dx = Math.abs(x - xi);
                int dy = Math.abs(y - yi);
                if (dx < cutoff && dy < cutoff && (dx + dy < bestDist)) {
                    bestDist = dx + dy;
                    bestIndex = i;
                }
            }
            selection = bestIndex;
            return bestIndex;
        }

        // Check if mouse click corresponds with existing envelope point (to
        // select point)
        private boolean checkSelection(int time, int amp) {
            int i = getSelection(time, amp);
            if (i < 0)
                return false;
            envTimes.setSelectedIndex(i / 2);
            envAmplitudes.setText(form.format(data.params.get(i + 1)));
            repaint();
            return true;
        }


        // parses an envelope data string and puts the data in the passed
        // vectors
        // returns null if there is no data to parse (null String)
        // Otherwise returns the value of the last time token so that MaxTime
        // can be set as necessary by caller
        public String parseEnvData(String envData, EnvData data) {
            if (envData == null)
                return null;
            // trim the open and close parens from envData
            int startx = envData.indexOf("(") + 1; // if no open paren, startx
                                                   // will be 0
            int endx = envData.indexOf(")");
            if (endx < 0)
                endx = envData.length();
            envData = envData.substring(startx, endx);
            StringTokenizer st = new StringTokenizer(envData);
            String type = st.nextToken();
            boolean valueType = type.endsWith("v");
            double defaultAmp = type.toLowerCase().startsWith("pwl") ? 
                                0.0 : 1.0;
            int i = 0;
            // pretend mouse is down to avoid making each point undo-able
            boolean save = mouseDown;
            mouseDown = true;
            String token1 = "1.0", token2;
            double time, amp;

            if (valueType) { // first element is value
                amp = defaultAmp;
                if (st.countTokens() >= 1) {
                    amp = Double.parseDouble(st.nextToken().trim());
                }
                data.params.add(0.0);
                data.params.add(amp);
                i++;
            }

            while (st.countTokens() >= 2) {
                token1 = st.nextToken().trim();
                time = Double.parseDouble(token1);
                token2 = st.nextToken().trim();
                amp = Double.parseDouble(token2);
                // System.out.println("time " + token1 + " amp " + token2 +
                // " size " + times.size());
                data.params.add(time);
                data.params.add(amp);
            }

            if (valueType && data.params.size() < 3) {
                data.params.add(1.0);
                data.params.add(defaultAmp);
            }

            mouseDown = save; // restore the mouseDown state
            if (!valueType) { // last element is time
                if (st.countTokens() > 0) {
                    token1 = st.nextToken().trim();
                }
            }
            data.maxT = Double.parseDouble(token1.trim());
            return token1;
        }


        // determine if this is PWL or PWE. This should really be
        // determined in parseEnvData, but since parseEnvData already
        // returns a string (the max time value), and because it is
        // not necessary to reparse everything, a separate function
        // is provided here:
        public boolean isEnvDataLinear(String envData) {
            int posn = envData.indexOf("pw"); // look for pwl or pwe
            boolean linear = true; // default in case of error
            // just in case text is upper case, check for PWL or PWE:
            if (posn < 0) {
                posn = envData.indexOf("PW");
            }
            if (posn >= 0) { // found pw or PW
                if (envData.length() > posn + 2) { // L or E exists
                    char c = envData.charAt(posn + 2);
                    linear = (c == 'l' || c == 'L');
                }
            }
            return linear;
        }


        // parse envelope from string and prepare to edit
        public void setEnv(String envData) {
            System.out.println("setEnv: envData " + envData);
            if (envData == null)
                return;
            boolean save = mouseDown;
            mouseDown = true;
            Vector<Double> params = new Vector<Double>();
            data.params.clear();
            String max = parseEnvData(envData, data);
            if (max != null) {
                maxT.setText(max);
            }
            mouseDown = save; // restore the mouseDown state
            System.out.println("  params " + params);
            // calculateDraws(true);
            modified = false;
            System.out.println("modified set to false in setEnv\n");
            repaint();
        }

        public void keyPressed(KeyEvent event) {
            // Graphics2D drawArea = image.createGraphics();
            if (event.getKeyCode() == KeyEvent.VK_DELETE) {
                deleteSelection();
                repaint();
            }
        }

        // fill rest of mouse functions
        public void mouseEntered(MouseEvent e) {
        }

        public void mouseExited(MouseEvent e) {
        }

        public void mouseClicked(MouseEvent e) {
        }

        public void mouseMoved(MouseEvent e) {
        }

        public void keyReleased(KeyEvent event) {
        }

        public void keyTyped(KeyEvent event) {
        }
    }

    private void saveHistory() {
        canvas.history.save(canvas.data);
    }

}
