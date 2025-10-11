namespace app;

partial class MainForm
{
    /// <summary>
    ///  Required designer variable.
    /// </summary>
    private System.ComponentModel.IContainer components = null!;

    private System.Windows.Forms.TableLayoutPanel mainLayout;
    private System.Windows.Forms.TableLayoutPanel headerTable;
    private System.Windows.Forms.Label modelLabel;
    private System.Windows.Forms.ComboBox modelComboBox;
    private System.Windows.Forms.Label modelPathLabel;
    private System.Windows.Forms.Button addVideosButton;
    private System.Windows.Forms.Button removeSelectedButton;
    private System.Windows.Forms.Button clearFinishedButton;
    private System.Windows.Forms.Button postprocessButton;
    private System.Windows.Forms.DataGridView queueGrid;
    private System.Windows.Forms.GroupBox logGroupBox;
    private System.Windows.Forms.TextBox logTextBox;

    /// <summary>
    ///  Clean up any resources being used.
    /// </summary>
    /// <param name="disposing">true if managed resources should be disposed; otherwise, false.
    /// </param>
    protected override void Dispose(bool disposing)
    {
        if (disposing && components != null)
        {
            components.Dispose();
        }
        base.Dispose(disposing);
    }

    #region Windows Form Designer generated code

    private void InitializeComponent()
    {
        mainLayout = new TableLayoutPanel();
        headerTable = new TableLayoutPanel();
        modelLabel = new Label();
        modelComboBox = new ComboBox();
        modelPathLabel = new Label();
        addVideosButton = new Button();
        removeSelectedButton = new Button();
        clearFinishedButton = new Button();
        postprocessButton = new Button();
        queueGrid = new DataGridView();
        logGroupBox = new GroupBox();
        logTextBox = new TextBox();
        mainLayout.SuspendLayout();
        headerTable.SuspendLayout();
        ((System.ComponentModel.ISupportInitialize)queueGrid).BeginInit();
        logGroupBox.SuspendLayout();
        SuspendLayout();
        // 
        // mainLayout
        // 
        mainLayout.ColumnCount = 1;
        mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
        mainLayout.Controls.Add(headerTable, 0, 0);
        mainLayout.Controls.Add(queueGrid, 0, 1);
        mainLayout.Controls.Add(logGroupBox, 0, 2);
        mainLayout.Dock = DockStyle.Fill;
        mainLayout.Location = new Point(0, 0);
        mainLayout.Name = "mainLayout";
        mainLayout.RowCount = 3;
        mainLayout.RowStyles.Add(new RowStyle());
        mainLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));
        mainLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 160F));
        mainLayout.Size = new Size(1040, 640);
        mainLayout.TabIndex = 0;
        // 
        // headerTable
        // 
        headerTable.AutoSize = true;
        headerTable.ColumnCount = 7;
        headerTable.ColumnStyles.Add(new ColumnStyle());
        headerTable.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 220F));
        headerTable.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
        headerTable.ColumnStyles.Add(new ColumnStyle());
        headerTable.ColumnStyles.Add(new ColumnStyle());
        headerTable.ColumnStyles.Add(new ColumnStyle());
        headerTable.ColumnStyles.Add(new ColumnStyle());
        headerTable.Controls.Add(modelLabel, 0, 0);
        headerTable.Controls.Add(modelComboBox, 1, 0);
        headerTable.Controls.Add(modelPathLabel, 2, 0);
        headerTable.Controls.Add(addVideosButton, 3, 0);
        headerTable.Controls.Add(removeSelectedButton, 4, 0);
        headerTable.Controls.Add(clearFinishedButton, 5, 0);
        headerTable.Controls.Add(postprocessButton, 6, 0);
        headerTable.Dock = DockStyle.Fill;
        headerTable.Location = new Point(3, 3);
        headerTable.Name = "headerTable";
        headerTable.RowCount = 1;
        headerTable.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));
        headerTable.Size = new Size(1034, 40);
        headerTable.TabIndex = 0;
        // 
        // modelLabel
        // 
        modelLabel.Anchor = AnchorStyles.Left;
        modelLabel.AutoSize = true;
        modelLabel.Location = new Point(3, 12);
        modelLabel.Name = "modelLabel";
        modelLabel.Size = new Size(44, 15);
        modelLabel.TabIndex = 0;
        modelLabel.Text = "Model:";
        // 
        // modelComboBox
        // 
        modelComboBox.Dock = DockStyle.Fill;
        modelComboBox.DropDownStyle = ComboBoxStyle.DropDownList;
        modelComboBox.FormattingEnabled = true;
        modelComboBox.Location = new Point(53, 6);
        modelComboBox.Margin = new Padding(3, 6, 3, 6);
        modelComboBox.Name = "modelComboBox";
        modelComboBox.Size = new Size(214, 23);
        modelComboBox.TabIndex = 1;
        modelComboBox.SelectedIndexChanged += modelComboBox_SelectedIndexChanged;
        // 
        // modelPathLabel
        // 
        modelPathLabel.AutoEllipsis = true;
        modelPathLabel.Dock = DockStyle.Fill;
        modelPathLabel.Location = new Point(273, 0);
        modelPathLabel.Name = "modelPathLabel";
        modelPathLabel.Size = new Size(393, 40);
        modelPathLabel.TabIndex = 2;
        modelPathLabel.TextAlign = ContentAlignment.MiddleLeft;
        // 
        // addVideosButton
        // 
        addVideosButton.AutoSize = true;
        addVideosButton.Location = new Point(672, 6);
        addVideosButton.Margin = new Padding(3, 6, 3, 6);
        addVideosButton.Name = "addVideosButton";
        addVideosButton.Size = new Size(79, 27);
        addVideosButton.TabIndex = 3;
        addVideosButton.Text = "Add Videos";
        addVideosButton.UseVisualStyleBackColor = true;
        addVideosButton.Click += addVideosButton_Click;
        // 
        // removeSelectedButton
        // 
        removeSelectedButton.AutoSize = true;
        removeSelectedButton.Location = new Point(757, 6);
        removeSelectedButton.Margin = new Padding(3, 6, 3, 6);
        removeSelectedButton.Name = "removeSelectedButton";
        removeSelectedButton.Size = new Size(108, 27);
        removeSelectedButton.TabIndex = 4;
        removeSelectedButton.Text = "Remove Pending";
        removeSelectedButton.UseVisualStyleBackColor = true;
        removeSelectedButton.Click += removeSelectedButton_Click;
        // 
        // clearFinishedButton
        // 
        clearFinishedButton.AutoSize = true;
        clearFinishedButton.Location = new Point(871, 6);
        clearFinishedButton.Margin = new Padding(3, 6, 3, 6);
        clearFinishedButton.Name = "clearFinishedButton";
        clearFinishedButton.Size = new Size(95, 27);
        clearFinishedButton.TabIndex = 5;
        clearFinishedButton.Text = "Clear Finished";
        clearFinishedButton.UseVisualStyleBackColor = true;
        clearFinishedButton.Click += clearFinishedButton_Click;
        // 
        // postprocessButton
        // 
        postprocessButton.AutoSize = true;
        postprocessButton.Location = new Point(972, 6);
        postprocessButton.Margin = new Padding(3, 6, 3, 6);
        postprocessButton.Name = "postprocessButton";
        postprocessButton.Size = new Size(59, 27);
        postprocessButton.TabIndex = 6;
        postprocessButton.Text = "Options";
        postprocessButton.UseVisualStyleBackColor = true;
        postprocessButton.Click += postprocessButton_Click;
        // 
        // queueGrid
        // 
        queueGrid.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize;
        queueGrid.Dock = DockStyle.Fill;
        queueGrid.Location = new Point(3, 49);
        queueGrid.Name = "queueGrid";
        queueGrid.RowHeadersVisible = false;
        queueGrid.Size = new Size(1034, 428);
        queueGrid.TabIndex = 1;
        // 
        // logGroupBox
        // 
        logGroupBox.Controls.Add(logTextBox);
        logGroupBox.Dock = DockStyle.Fill;
        logGroupBox.Location = new Point(3, 483);
        logGroupBox.Name = "logGroupBox";
        logGroupBox.Size = new Size(1034, 154);
        logGroupBox.TabIndex = 2;
        logGroupBox.TabStop = false;
        logGroupBox.Text = "Activity Log";
        // 
        // logTextBox
        // 
        logTextBox.Dock = DockStyle.Fill;
        logTextBox.Location = new Point(3, 19);
        logTextBox.Multiline = true;
        logTextBox.Name = "logTextBox";
        logTextBox.ReadOnly = true;
        logTextBox.ScrollBars = ScrollBars.Vertical;
        logTextBox.Size = new Size(1028, 132);
        logTextBox.TabIndex = 0;
        // 
        // MainForm
        // 
        AllowDrop = true;
        AutoScaleDimensions = new SizeF(7F, 15F);
        AutoScaleMode = AutoScaleMode.Font;
        ClientSize = new Size(1040, 640);
        Controls.Add(mainLayout);
        MinimumSize = new Size(900, 600);
        Name = "MainForm";
        StartPosition = FormStartPosition.CenterScreen;
        Text = "DeepFunGen";
        DragDrop += MainForm_DragDrop;
        DragEnter += MainForm_DragEnter;
        mainLayout.ResumeLayout(false);
        mainLayout.PerformLayout();
        headerTable.ResumeLayout(false);
        headerTable.PerformLayout();
        ((System.ComponentModel.ISupportInitialize)queueGrid).EndInit();
        logGroupBox.ResumeLayout(false);
        logGroupBox.PerformLayout();
        ResumeLayout(false);
    }

    #endregion
}
